import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from fpsrg_vector_vllm.artifact import Artifact, sha256
from fpsrg_vector_vllm.routing import token_routes
from fpsrg_vector_vllm.model import layer_banks
from fpsrg_vector_vllm.server import Service, expand, eos_token_ids, parse_completion
from fpsrg_vector_vllm.precision import adapter_target


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.axes=list('OCEANRTVMS')
        self.nodes=[dict(node_id=i,persona_id=i+1,scores={a:30.+40*i for a in self.axes},
                         coefficients={a:(-1 if i==0 else 1) for a in self.axes}) for i in range(2)]
        (self.root/'catalog.json').write_text(json.dumps(dict(nodes=self.nodes)))
        vectors=np.arange(10*5120,dtype=np.float32).reshape(10,5120)/np.float32(123456)
        np.savez(self.root/'vectors.npz',vectors=vectors)
        self.spec=dict(schema_version=1,artifact_id='test-ten',backend='ten_off',
            model=dict(path='weights',fingerprint='a'*64),adapter=dict(path='adapter',commit_sha256='b'*64),
            axes=self.axes,layer_by_trait=dict(zip(self.axes,[40,32,32,16,32,32,40,32,40,32])),
            assets={key:dict(path=key+'.'+ext,sha256=sha256(self.root/(key+'.'+ext)))
                    for key,ext in [('catalog','json'),('vectors','npz')]},generation=dict(thinking=False))
        self.write()
    def tearDown(self): self.temp.cleanup()
    def write(self):
        (self.root/'manifest.json').write_text(json.dumps(self.spec))
    def artifact(self): return Artifact(self.root/'manifest.json')
    def service(self,n=5):
        service=Service(self.artifact());service.ready=True
        service.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:'x<think>\n\n</think>\n\n',encode=lambda *a,**kw:[1]*n)
        return service
    def request(self,**overrides):
        return dict(node_id=0,messages=[dict(role='user',content='你好')],max_tokens=64,**overrides)


class CatalogTests(Fixture):
    def test_read_and_nearest(self):
        art=self.artifact(); self.assertEqual(art.choose(targets={a:65 for a in self.axes})['node_id'],1)
        self.assertEqual(art.choose(targets={a:50 for a in self.axes})['node_id'],0)
    def test_explicit_persona_mapping(self):
        art=self.artifact();self.assertEqual(art.route(art.choose(node_id=0)),1)
    def test_bad_node_or_target(self):
        art=self.artifact()
        for kwargs in [dict(node_id=True),dict(node_id=3),dict(node_id=0,targets={}),dict(targets={'A':50}),dict(targets={a:float('nan') for a in self.axes})]:
            with self.assertRaises(ValueError):art.choose(**kwargs)
    def test_changed_asset_is_rejected(self):
        (self.root/'catalog.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA256'):self.artifact()
    def test_unresolved_path(self):
        self.spec['model']['path']='${FPSRG_MISSING_VARIABLE}/model';self.write()
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ValueError,'Unresolved'):self.artifact()
    def test_cache_is_persona_and_model_isolated(self):
        art=self.artifact();old=art.cache_salt(1)
        self.assertNotEqual(old,art.cache_salt(2))
        self.spec['model']['fingerprint']='c'*64;self.write()
        self.assertNotEqual(old,self.artifact().cache_salt(1))
    def test_no_emotion_in_ten(self):
        art=self.artifact()
        with self.assertRaises(ValueError):art.route(art.nodes[0],'anger')
    def test_compact_layer_sum_matches_direct_fp32(self):
        art=self.artifact();capacity,banks=layer_banks(art);self.assertEqual(capacity,3)
        with np.load(art.asset('vectors')) as z:vectors=z['vectors']
        for layer,(bank,lookup) in banks.items():
            self.assertFalse(bank[0].any())
            for node in art.nodes:
                expected=np.zeros(5120,np.float32)
                for i,a in enumerate(art.axes):
                    if art.spec['layer_by_trait'][a]==layer:
                        expected=expected+vectors[i]*np.float32(node['coefficients'][a])
                np.testing.assert_array_equal(bank[lookup[node['persona_id']]],expected)


class ProtocolTests(Fixture):
    def test_nonempty_prompt_and_seed(self):
        value=self.service().prepare(self.request(seed=9))
        self.assertEqual(value['seed'],9); self.assertEqual(value['persona'],1)
    def test_thinking_mismatch(self):
        with self.assertRaises(ValueError):self.service().prepare(self.request(enable_thinking=True))
    def test_no_silent_truncation(self):
        self.service(32704).prepare(self.request())
        with self.assertRaisesRegex(ValueError,'Context'):self.service(32705).prepare(self.request())
    def test_bad_fields(self):
        for kw in [dict(seed=True),dict(temperature=float('nan')),dict(stream=True),dict(model='unknown'),dict(emotion='anger')]:
            with self.assertRaises(ValueError):self.service().prepare(self.request(**kw))
    def test_system_once(self):
        req=self.request(system='Hello');req['messages'].insert(0,dict(role='system',content='second'))
        with self.assertRaises(ValueError):self.service().prepare(req)
    def test_atomic_batch(self):
        service=self.service();bad=self.request();bad['node_id']=-1
        with self.assertRaises(ValueError):service.submit([self.request(),bad])
        self.assertTrue(service.pending.empty())
    def test_a3_b5_different_prompts(self):
        data=dict(defaults=dict(max_tokens=32),groups=[
            dict(name='A',node_id=0,requests=[dict(messages=[dict(role='user',content=str(i))]) for i in range(3)]),
            dict(name='B',node_id=1,n=5,messages=[dict(role='user',content='B prompt')])])
        rows=expand(data);self.assertEqual(len(rows),8)
        self.assertEqual(rows[1][1]['messages'][0]['content'],'1');self.assertEqual(rows[4][1]['node_id'],1)
        self.assertEqual(len(self.service().submit([item for _,item in rows])),8)
    def test_invalid_batch(self):
        for value in [dict(groups=[]),dict(groups=[dict(n=65)]),dict(requests=[],groups=[]),dict(groups=[dict(n=2,requests=[{}])])]:
            with self.assertRaises(ValueError):expand(value)


class RoutingTests(unittest.TestCase):
    def test_post_reorder_chunked_prefill_decode_padding(self):
        value=token_routes(['decode','other','prefill'],dict(prefill=4,decode=1,other=2),
                           dict(prefill=1,decode=2,other=0),10,7,3)
        self.assertEqual(value.tolist(),[2,0,0,1,1,1,1,0,0,0])
    def test_padding_reset(self):
        out=np.full(8,9,np.int64)
        token_routes(['a'],{'a':1},{'a':2},8,1,3,out)
        self.assertEqual(out.tolist(),[2,0,0,0,0,0,0,0])
    def test_unknown_request_rejected(self):
        with self.assertRaises(ValueError):token_routes(['a'],{'a':2,'b':1},{'a':1},3,3,3)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):token_routes(['a','a'],{'a':1},{'a':1},2,2,3)
    def test_large_ten_catalog_routing(self):
        self.assertEqual(token_routes(['a'],{'a':1},{'a':59049},1,1,59050).tolist(),[59049])
    def test_adapter_stacked_projection(self):
        self.assertEqual(adapter_target('base_model.model.model.language_model.layers.3.self_attn.k_proj.lora_A.weight'),
                         ('language_model.model.layers.3.self_attn.qkv_proj',1,'A'))


class EmotionTests(Fixture):
    def setUp(self):
        super().setUp();axes=list('ACNRMT')
        nodes=[dict(node_id=42,persona_id=1,scores={a:50 for a in axes},coefficients={a:0 for a in axes})]
        (self.root/'catalog.json').write_text(json.dumps(dict(nodes=nodes)))
        bank=np.zeros((6,1,5120),np.float32);bank[2:4,:,0]=1;bank[4:,:,0]=-1
        np.savez(self.root/'bank.npz',bank=bank,layers=np.array([24],np.int64))
        self.spec.update(artifact_id='test-six',backend='huihui6_full',axes=axes,adapter=None,
            generation=dict(thinking=True),emotion=dict(base_rows=2,modes=['none','joy','anger'],strength=2.0),
            assets={k:dict(path=k+'.'+ext,sha256=sha256(self.root/(k+'.'+ext))) for k,ext in [('catalog','json'),('bank','npz')]})
        self.write()
    def test_nonsequential_node_to_bank(self):
        art=self.artifact();node=art.choose(node_id=42)
        self.assertEqual([art.route(node,m) for m in ['none','joy','anger']],[1,3,5])
    def test_emotion_cache_isolation(self):
        art=self.artifact();self.assertEqual(len({art.cache_salt(art.route(art.nodes[0],m)) for m in ['none','joy','anger']}),3)
    def test_bank_roundtrip(self):
        capacity,banks=layer_banks(self.artifact());self.assertEqual(capacity,6)
        self.assertEqual(banks[24][0][5,0],-1)


class CompletionTests(unittest.TestCase):
    def tokenizer(self):
        return SimpleNamespace(eos_token_id=0,decode=lambda ids,**kw:''.join({0:'<eos>',1:'<end>'}.get(v,chr(v)) for v in ids))
    def completion(self,text,ending=(1,),finish='stop'):
        return SimpleNamespace(token_ids=[ord(c) for c in text]+list(ending),finish_reason=finish)
    def test_multiple_native_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'generation_config.json').write_text('{"eos_token_id":[0,1]}')
            self.assertEqual(eos_token_ids(self.tokenizer(),root),{0,1})
        value=parse_completion(self.tokenizer(),self.completion('我想好了。</think>好。'),True,{0,1})
        self.assertTrue(value['complete']);self.assertEqual(value['answer'],'好。')
    def test_no_fabricated_termination(self):
        for output in [self.completion('思考</think>好',ending=(),finish='length'),
                       self.completion('没有结束思考'),self.completion('</think>好'),
                       self.completion('思考</think>'),self.completion('思考</think>好</think>')]:
            self.assertFalse(parse_completion(self.tokenizer(),output,True,{0,1})['complete'])
    def test_off_eos_removed_only_from_answer(self):
        value=parse_completion(self.tokenizer(),self.completion('你好'),False,{0,1})
        self.assertTrue(value['complete']);self.assertEqual(value['answer'],'你好');self.assertIn('<end>',value['raw'])


if __name__=='__main__':unittest.main()
