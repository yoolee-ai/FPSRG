"""Single-GPU continuous-batch HTTP service with per-request persona and prompt."""
import argparse
import concurrent.futures as futures
import hmac
import importlib.metadata
import json
import math
import os
import queue
import secrets
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .artifact import Artifact


def eos_token_ids(tokenizer, model_path):
    values=[getattr(tokenizer,'eos_token_id',None),
            json.loads((model_path/'generation_config.json').read_text()).get('eos_token_id')]
    result=set()
    for value in values:
        if type(value) is int: result.add(value)
        elif isinstance(value,list): result.update(value)
    if not result or any(type(value) is not int for value in result):
        raise RuntimeError('Valid native EOS IDs required')
    return result


def parse_completion(tokenizer, completion, thinking, eos_ids):
    ids=list(completion.token_ids)
    if any(type(value) is not int for value in ids): raise RuntimeError('Noninteger generated token ID')
    raw=tokenizer.decode(ids,skip_special_tokens=False)
    ended=bool(ids) and ids[-1] in eos_ids
    body=tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False)
    if thinking:
        thought,separator,answer=body.partition('</think>')
        closed=bool(separator) and body.count('</think>')==1
        if not closed: answer=''
    else: thought,closed,answer='',True,body
    complete=(completion.finish_reason=='stop' and ended and closed and bool(answer.strip())
              and not any(token in eos_ids for token in ids[:-1])
              and (not thinking or bool(thought.strip()))
              and '<think>' not in answer and '</think>' not in answer)
    return dict(raw=raw,thought=thought.strip(),answer=answer.strip(),closed=closed,ended=ended,complete=complete)


def expand(data):
    if not isinstance(data,dict): raise ValueError('Expected JSON object')
    if 'requests' in data:
        items=data['requests']
        if set(data)!={'requests'} or not isinstance(items,list) or not 1<=len(items)<=64:
            raise ValueError('Provide 1–64 requests')
        return [(dict(group_index=i,sample_index=0),item) for i,item in enumerate(items)]
    if set(data)-{'defaults','groups'}: raise ValueError('Use requests OR defaults/groups')
    defaults=data.get('defaults',{}); groups=data.get('groups',[]); result=[]
    if not isinstance(defaults,dict) or not isinstance(groups,list): raise ValueError('Invalid groups')
    for g,group in enumerate(groups):
        if not isinstance(group,dict): raise ValueError('Invalid group')
        opts=dict(defaults,**{k:v for k,v in group.items() if k not in ('name','n','requests')})
        if 'requests' in group:
            items=group['requests']
            if 'n' in group or not isinstance(items,list) or not items: raise ValueError('Use requests OR n')
        else:
            n=group.get('n',1)
            if type(n) is not int or not 1<=n<=64: raise ValueError('n must be 1–64')
            items=[{} for _ in range(n)]
        for index,item in enumerate(items):
            if not isinstance(item,dict): raise ValueError('Invalid request override')
            result.append((dict(group_index=g,group_name=group.get('name',str(g)),sample_index=index),dict(opts,**item)))
            if len(result)>64: raise ValueError('At most 64 requests')
    if not result: raise ValueError('No requests')
    return result


class Service:
    def __init__(self,artifact,max_active=16,memory_fraction=.90):
        self.artifact=artifact; self.max_active=max_active; self.memory_fraction=memory_fraction
        self.pending=queue.Queue(64); self.lock=threading.Lock(); self.ready=False; self.error=None
        self.tokenizer=None; self.active={}; self.completed=0; self.stop=False

    def prepare(self,data):
        if not self.ready: raise RuntimeError('Model loading/unavailable')
        allowed={'model','messages','system','targets','node_id','emotion','temperature','max_tokens','seed','stream','enable_thinking'}
        if not isinstance(data,dict) or set(data)-allowed: raise ValueError('Unknown request fields')
        if data.get('model','grid') not in ('grid',self.artifact.spec['artifact_id']) or data.get('stream',False) is not False:
            raise ValueError('This endpoint serves one artifact; stream=false required')
        cfg=self.artifact.generation; thinking=data.get('enable_thinking',cfg['thinking'])
        if type(thinking) is not bool or thinking != cfg['thinking']: raise ValueError('Thinking mode differs from artifact')
        cap=data.get('max_tokens',cfg['max_tokens']); temp=data.get('temperature',.7); seed=data.get('seed',secrets.randbits(32))
        if type(cap) is not int or not 1<=cap<=16384: raise ValueError('max_tokens must be 1–16384')
        if type(temp) not in (int,float) or not math.isfinite(temp) or not 0<=temp<=2: raise ValueError('Invalid temperature')
        if type(seed) is not int or not 0<=seed<2**63: raise ValueError('Invalid seed')
        messages=data.get('messages')
        if not isinstance(messages,list) or not 1<=len(messages)<=128: raise ValueError('1–128 messages required')
        for i,m in enumerate(messages):
            if (not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in ('system','user','assistant')
                or not isinstance(m['content'],str) or not m['content'].strip() or (m['role']=='system' and i!=0)):
                raise ValueError('Only nonempty text system/user/assistant messages are supported')
        if messages[-1]['role']!='user': raise ValueError('Last message must be user')
        messages=[dict(m) for m in messages]
        if 'system' in data and messages[0]['role']=='system': raise ValueError('Provide system once')
        if messages[0]['role']!='system':
            system=data.get('system',cfg['system'])
            if not isinstance(system,str) or not system.strip(): raise ValueError('Invalid system')
            messages.insert(0,dict(role='system',content=system))
        node=self.artifact.choose(data.get('node_id'),data.get('targets'))
        emotion=data.get('emotion','none'); persona=self.artifact.route(node,emotion)
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                enable_thinking=thinking,reasoning_effort='medium',preserve_thinking=False,tools=None)
        suffix='<think>\n' if thinking else '<think>\n\n</think>\n\n'
        if not text.endswith(suffix): raise RuntimeError('Native chat-template suffix mismatch')
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids): raise RuntimeError('Invalid token IDs')
        if len(ids)+cap>cfg['max_model_len']: raise ValueError('Context overflow; no silent truncation')
        return dict(id='chatcmpl-'+uuid.uuid4().hex,ids=ids,node=node,persona=persona,emotion=emotion,
                    thinking=thinking,cap=cap,temp=float(temp),seed=seed,submitted=time.monotonic())

    def submit(self,items):
        with self.lock:
            if not self.ready: raise RuntimeError('Service unavailable')
            jobs=[self.prepare(item) for item in items]
            if self.pending.qsize()+len(jobs)>64: raise OverflowError('Queue full')
            fs=[futures.Future() for _ in jobs]
            for job,future in zip(jobs,fs): self.pending.put_nowait((job,future))
        return fs

    def run(self):
        try:
            expected={'vllm':'0.19.0','torch':'2.10.0','transformers':'5.15.1'}
            for package,version in expected.items():
                if importlib.metadata.version(package).split('+')[0]!=version:
                    raise RuntimeError('Unsupported runtime version: '+package+' (expected '+version+')')
            os.environ['FPSRG_ARTIFACT_MANIFEST']=str(self.artifact.path)
            from . import model
            model.install_runtime()
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(self.artifact.model,local_files_only=True)
            eos_ids=eos_token_ids(self.tokenizer,self.artifact.model)
            llm=LLM(model=str(self.artifact.model),tokenizer=str(self.artifact.model),dtype='bfloat16',
                trust_remote_code=False,tensor_parallel_size=1,language_model_only=True,
                max_model_len=self.artifact.generation['max_model_len'],max_num_seqs=self.max_active,
                max_num_batched_tokens=2048,gpu_memory_utilization=self.memory_fraction,
                enable_chunked_prefill=True,enable_prefix_caching=True,enforce_eager=True,
                compilation_config=dict(mode=0,cudagraph_mode='NONE',cudagraph_capture_sizes=[]),
                worker_extension_cls='fpsrg_vector_vllm.model.PersonaWorker',async_scheduling=False,
                disable_log_stats=True,seed=20261104)
            engine=llm.llm_engine
            self.audit=llm.collective_rpc('persona_audit')
            self.ready=True
            while not self.stop:
                while len(self.active)<self.max_active:
                    try: job,future=self.pending.get_nowait()
                    except queue.Empty: break
                    if future.cancelled(): continue
                    try:
                        engine.add_request(job['id'],dict(prompt_token_ids=job['ids'],
                            cache_salt=self.artifact.cache_salt(job['persona'])),
                            SamplingParams(temperature=job['temp'],max_tokens=job['cap'],seed=job['seed'],
                                top_p=1.,top_k=-1,skip_special_tokens=False,spaces_between_special_tokens=False,
                                extra_args={'fpsrg_persona':job['persona']}))
                    except BaseException as exc:
                        future.set_exception(RuntimeError('Request admission failed')); raise exc
                    self.active[job['id']]=(job,future)
                if not self.active: time.sleep(.005); continue
                for rid,(job,future) in list(self.active.items()):
                    if future.cancelled() or time.monotonic()-job['submitted']>1800:
                        engine.abort_request([rid]); del self.active[rid]
                        if not future.done(): future.set_exception(TimeoutError('Request deadline'))
                for out in engine.step():
                    if not out.finished: continue
                    job,future=self.active.pop(out.request_id)
                    value=out.outputs[0]
                    parsed=parse_completion(self.tokenizer,value,job['thinking'],eos_ids)
                    result=dict(id=job['id'],object='chat.completion',model=self.artifact.spec['artifact_id'],
                        choices=[dict(index=0,message=dict(role='assistant',content=parsed['answer'],reasoning_content=parsed['thought']),
                                      finish_reason=value.finish_reason)],grid=job['node'],
                        usage=dict(prompt_tokens=len(job['ids']),completion_tokens=len(value.token_ids),total_tokens=len(job['ids'])+len(value.token_ids)),
                        metadata=dict(artifact_id=self.artifact.spec['artifact_id'],enable_thinking=job['thinking'],
                            thinking_closed=parsed['closed'],complete=parsed['complete'],ended_with_eos=parsed['ended'],
                            emotion=job['emotion'],emotion_strength=0. if job['emotion']=='none' else 2.,
                            measured_after_overlay=False,runtime_persona_id=job['persona'],
                            cache_salt=self.artifact.cache_salt(job['persona']),cached_tokens=out.num_cached_tokens,
                            raw_generated=parsed['raw'],finish_reason=value.finish_reason,total_seconds=time.monotonic()-job['submitted'],
                            score_warning='Stored catalog scores are not a new measurement of this conversation.'))
                    if not future.done(): future.set_result(result)
                    self.completed+=1
        except BaseException as exc:
            self.ready=False; self.error=type(exc).__name__+': '+str(exc)
            for _,future in self.active.values():
                if not future.done(): future.set_exception(RuntimeError('Service failed'))
            with self.lock:
                while not self.pending.empty():
                    _,future=self.pending.get_nowait()
                    if not future.done(): future.set_exception(RuntimeError('Service failed'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True); parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--bind',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--max-active',type=int,default=16); parser.add_argument('--memory-fraction',type=float,default=.90)
    args=parser.parse_args()
    token=os.environ.get('FPSRG_API_TOKEN','')
    if args.bind not in ('127.0.0.1','::1','localhost') and not token:
        parser.error('Set FPSRG_API_TOKEN before binding beyond loopback')
    if not 1<=args.max_active<=64 or not .1<=args.memory_fraction<=.95:
        parser.error('max-active must be 1–64 and memory-fraction .1–.95')
    peers=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if peers: raise RuntimeError('GPU is occupied; no existing process is stopped')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',FLA_CACHE_MODE='disabled',NLA_CONV_BACKEND='fla')
    artifact=Artifact(args.manifest); service=Service(artifact,args.max_active,args.memory_fraction)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,value,status=200):
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def auth(self):
            return not token or hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def do_GET(self):
            if self.path=='/health': return self.send(dict(ready=service.ready,error=service.error))
            if not self.auth(): return self.send(dict(error='Unauthorized'),401)
            if self.path=='/catalog': return self.send(artifact.public())
            if self.path=='/metrics': return self.send(dict(active=len(service.active),pending=service.pending.qsize(),completed=service.completed))
            return self.send(dict(error='Not found'),404)
        def do_POST(self):
            if not self.auth(): return self.send(dict(error='Unauthorized'),401)
            if self.path not in ('/v1/chat/completions','/v1/chat/completions/batch'): return self.send(dict(error='Not found'),404)
            fs=[]
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4194304: raise ValueError('Body must be 1 byte–4 MiB')
                self.connection.settimeout(30); data=json.loads(self.rfile.read(length))
                batch=self.path.endswith('/batch'); labeled=expand(data) if batch else [(None,data)]
                fs=service.submit([item for _,item in labeled]); deadline=time.monotonic()+1810
                rows=[dict(label or {},completion=f.result(timeout=max(.1,deadline-time.monotonic()))) for (label,_),f in zip(labeled,fs)]
                self.send(dict(object='grid.batch',count=len(rows),results=rows) if batch else rows[0]['completion'])
            except (ValueError,TypeError,KeyError) as exc: self.send(dict(error=str(exc)),400)
            except OverflowError: self.send(dict(error='Queue full'),429)
            except TimeoutError: self.send(dict(error='Deadline exceeded'),504)
            except RuntimeError: self.send(dict(error='Service unavailable'),503)
            except (BrokenPipeError,ConnectionResetError): pass
            finally:
                for f in fs:
                    if not f.done(): f.cancel()
    server=ThreadingHTTPServer((args.bind,args.port),Handler); server.daemon_threads=True
    threading.Thread(target=service.run,daemon=True).start()
    try: server.serve_forever()
    finally: service.stop=True; service.ready=False; server.server_close()


if __name__=='__main__': main()
