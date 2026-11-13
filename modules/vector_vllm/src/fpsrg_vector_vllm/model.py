"""Version-pinned Qwen3.5/Huihui vLLM integration; eager TP1 only.

Routing follows vLLM's _preprocess batch reordering, covering both prefill and
decode. Precision functions are retained from the deployed implementation.
"""
import os
import json
import numpy as np
from .artifact import Artifact, sha256
from .routing import token_routes


def layer_banks(artifact):
    """Return compact per-layer FP32 banks and persona->bank-row lookups."""
    if artifact.backend == 'huihui6_full':
        with np.load(artifact.asset('bank'), allow_pickle=False) as z:
            bank=z['bank'].copy(); layers=z['layers'].copy()
        if (bank.dtype != np.float32 or bank.ndim != 3 or bank.shape[2] != 5120
            or not np.isfinite(bank).all() or layers.dtype != np.int64
            or layers.tolist() != sorted(set(layers.tolist()))
            or len(layers) != bank.shape[1] or np.any(layers<0) or np.any(layers>=64)
            or np.any(bank[0])):
            raise ValueError('Invalid FP32 Huihui bank/layers/zero row')
        capacity=bank.shape[0]
        for node in artifact.nodes:
            for mode in (artifact.emotion['modes'] if artifact.emotion else ['none']):
                if artifact.route(node,mode)>=capacity:
                    raise ValueError('Catalog persona exceeds bank')
        if artifact.emotion and capacity != artifact.emotion['base_rows']*3:
            raise ValueError('Emotion bank must contain none/joy/anger blocks')
        return capacity,{int(layer):(bank[:,col],np.arange(capacity,dtype=np.int64))
                         for col,layer in enumerate(layers) if np.any(bank[:,col])}
    with np.load(artifact.asset('vectors'),allow_pickle=False) as z:
        vectors=z['vectors'].copy()
    if vectors.shape!=(10,5120) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all():
        raise ValueError('Expected ten finite FP32 vectors')
    expected=dict(zip('OCEANRTVMS',[40,32,32,16,32,32,40,32,40,32]))
    if artifact.spec['layer_by_trait']!=expected:
        raise ValueError('ten_off layer recipe differs from the released calibration')
    capacity=max(n['persona_id'] for n in artifact.nodes)+1
    if capacity>1000000:
        raise ValueError('Unreasonably sparse persona IDs')
    seen={}; result={}
    for n in artifact.nodes:
        coeff=tuple(n['coefficients'][a] for a in artifact.axes)
        if any(v not in (-1,0,1) for v in coeff):
            raise ValueError('ten_off supports the original {-1,0,1} grid')
        if n['persona_id'] in seen and seen[n['persona_id']]!=coeff:
            raise ValueError('Conflicting persona mapping')
        if n['persona_id']==0 and any(coeff):
            raise ValueError('Persona row zero must have a zero vector')
        seen[n['persona_id']]=coeff
    for layer in sorted(set(expected.values())):
        indices=[i for i,a in enumerate(artifact.axes) if expected[a]==layer]
        unique=sorted({tuple(n['coefficients'][artifact.axes[i]] for i in indices) for n in artifact.nodes})
        rows=[np.zeros(5120,dtype=np.float32)]; codes={}
        for coeff in unique:
            delta=np.zeros(5120,dtype=np.float32)
            for i,alpha in zip(indices,coeff):
                if alpha: delta=delta+vectors[i]*np.float32(alpha)
            codes[coeff]=len(rows); rows.append(delta)
        lookup=np.zeros(capacity,dtype=np.int64)
        for n in artifact.nodes:
            lookup[n['persona_id']]=codes[tuple(n['coefficients'][artifact.axes[i]] for i in indices)]
        result[layer]=(np.stack(rows),lookup)
    return capacity,result


def install_routing():
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner,'_fpsrg_portable_installed',False): return
    if any(getattr(GPUModelRunner,k,False) for k in ('_fpsrg_installed','_acnr_routing_installed')):
        raise RuntimeError('Another vector-routing patch is already installed')
    original=GPUModelRunner._preprocess
    def preprocess(self,scheduler_output,num_input_tokens,*args,**kwargs):
        result=original(self,scheduler_output,num_input_tokens,*args,**kwargs)
        model=self.model
        if num_input_tokens>model.fpsrg_persona_ids.numel():
            raise RuntimeError('Token routing capacity exceeded')
        if not hasattr(self,'_fpsrg_portable_host'):
            self._fpsrg_portable_host=torch.zeros(model.fpsrg_persona_ids.numel(),dtype=torch.long,pin_memory=True)
        req_ids=list(self.input_batch.req_ids[:self.input_batch.num_reqs])
        personas={rid:(self.requests[rid].sampling_params.extra_args or {}).get('fpsrg_persona')
                  for rid in req_ids if scheduler_output.num_scheduled_tokens.get(rid,0)}
        token_routes(req_ids,scheduler_output.num_scheduled_tokens,personas,int(num_input_tokens),
                     int(scheduler_output.total_num_scheduled_tokens),model.fpsrg_capacity,
                     self._fpsrg_portable_host.numpy())
        model.fpsrg_persona_ids[:num_input_tokens].copy_(self._fpsrg_portable_host[:num_input_tokens],non_blocking=True)
        model.fpsrg_routing_calls+=1
        return result
    GPUModelRunner._preprocess=preprocess
    GPUModelRunner._fpsrg_portable_installed=True


def model_class():
    import torch
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from .precision import residual_intervention, _install_hf_same_rounding, _install_fixed_lora, COMMIT
    install_routing()
    class SteeredQwen35(Qwen3_5ForConditionalGeneration):
        def __init__(self,*,vllm_config,prefix='model'):
            art=Artifact(os.environ['FPSRG_ARTIFACT_MANIFEST'])
            if getattr(vllm_config,'lora_config',None) is not None:
                raise ValueError('Dynamic vLLM LoRA is not part of this precision recipe')
            if (vllm_config.parallel_config.tensor_parallel_size!=1
                or str(vllm_config.compilation_config.cudagraph_mode)!='NONE'):
                raise ValueError('This release requires eager TP1; CUDA Graphs are not admitted')
            if str(art.model)!=str(vllm_config.model_config.model):
                raise ValueError('Model path does not match artifact')
            if art.backend=='huihui6_full':
                commit=json.loads((art.model/'commit.json').read_text())
                if sha256(art.model/'commit.json')!=art.spec['model']['fingerprint'] or commit.get('lora') is not False or commit.get('epoch')!=2:
                    raise ValueError('Expected the full epoch-2 Huihui checkpoint')
            elif art.spec['adapter']['commit_sha256']!=COMMIT:
                raise ValueError('Incorrect ten-axis LoRA identity')
            super().__init__(vllm_config=vllm_config,prefix=prefix)
            self.fpsrg_artifact=art
            capacity,banks=layer_banks(art)
            device=self.language_model.model.embed_tokens.weight.device
            self.fpsrg_capacity=capacity; self.fpsrg_routing_calls=0
            self.register_buffer('fpsrg_persona_ids',torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens,
                                 dtype=torch.long,device=device),persistent=False)
            self.fpsrg_layers=sorted(banks)
            for layer,(bank,lookup) in banks.items():
                # All per-layer arithmetic has already happened in FP32. Round once.
                self.register_buffer(f'fpsrg_bank_{layer}',torch.from_numpy(np.ascontiguousarray(bank)).to(device=device,dtype=torch.bfloat16),persistent=False)
                self.register_buffer(f'fpsrg_lookup_{layer}',torch.from_numpy(lookup).to(device),persistent=False)
                def hook(module,args,output,layer=layer):
                    hidden,residual=output
                    if hidden.shape[0]>self.fpsrg_persona_ids.numel(): raise RuntimeError('Hidden-token overflow')
                    rows=getattr(self,f'fpsrg_lookup_{layer}')[self.fpsrg_persona_ids[:hidden.shape[0]]]
                    return residual_intervention(hidden,residual,getattr(self,f'fpsrg_bank_{layer}')[rows])
                self.language_model.model.layers[layer].register_forward_hook(hook)
        def load_weights(self,weights):
            loaded=super().load_weights(weights)
            os.environ['FPSRG_HF_SAME_ROUNDING']='1'
            if self.fpsrg_artifact.backend=='ten_off':
                os.environ['FPSRG_ADAPTER_CHECKPOINT']=str(self.fpsrg_artifact.resolve(self.fpsrg_artifact.spec['adapter']['path']))
                self.fpsrg_lora=_install_fixed_lora(self,torch)
            else: self.fpsrg_lora=None
            self.fpsrg_rounding=_install_hf_same_rounding(self,torch)
            self.language_model.lm_head.float()
            return loaded
        def compute_logits(self,hidden_states):
            return self.language_model.compute_logits(hidden_states.float())
    SteeredQwen35.__module__=__name__; SteeredQwen35.__qualname__='SteeredQwen35'
    return SteeredQwen35


def __getattr__(name):
    if name=='SteeredQwen35':
        value=model_class(); globals()[name]=value; return value
    raise AttributeError(name)


class PersonaWorker:
    def persona_audit(self):
        model=self.model_runner.model
        return dict(artifact_id=model.fpsrg_artifact.spec['artifact_id'],identity=model.fpsrg_artifact.identity,
                    routing_calls=model.fpsrg_routing_calls,layers=model.fpsrg_layers,
                    persona_capacity=model.fpsrg_capacity,lora=model.fpsrg_lora,rounding=model.fpsrg_rounding)


def install_runtime():
    from vllm import ModelRegistry
    install_routing()
    ModelRegistry.register_model('Qwen3_5ForConditionalGeneration','fpsrg_vector_vllm.model:SteeredQwen35')


if os.environ.get('FPSRG_ARTIFACT_MANIFEST'):
    install_runtime()
