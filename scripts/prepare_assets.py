"""Materialize the six-axis emotion bank from compact, checked components."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def prepare(root):
 root=Path(root);spec=json.loads((root/'bank_components.json').read_text())
 for filename,digest in spec['files'].items():
  if sha(root/filename)!=digest:raise ValueError('Component checksum mismatch: '+filename)
 dest=root/'bank.npz'
 if dest.exists():
  if sha(dest)!=spec['output_sha256']:raise ValueError('Existing bank checksum mismatch')
  return dest
 with np.load(root/'base_bank.npz',allow_pickle=False) as z:
  base=z['bank'];old_layers=z['layers'].tolist()
 with np.load(root/'emotion_vectors.npz',allow_pickle=False) as z:
  vectors={mode:z[mode].copy() for mode in ('joy','anger')}
 layers=spec['layers'];rows=spec['base_rows']
 bank=np.zeros((rows*3,len(layers),base.shape[-1]),dtype=np.float32)
 for mi,mode in enumerate(['none','joy','anger']):
  start=mi*rows
  for i,layer in enumerate(old_layers):bank[start:start+rows,layers.index(layer)]=base[:,i]
  if mode!='none':bank[start:start+rows,layers.index(spec['emotion_layers'][mode])]+=vectors[mode]*np.float32(2.)
 fd,tmp=tempfile.mkstemp(prefix='.bank-',suffix='.npz',dir=root)
 try:
  with os.fdopen(fd,'wb') as f:np.savez_compressed(f,bank=bank,layers=np.array(layers,dtype=np.int64))
  if sha(tmp)!=spec['output_sha256']:raise ValueError('Generated bank checksum mismatch')
  os.replace(tmp,dest)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 return dest

if __name__=='__main__':
 root=Path(os.environ.get('FPSRG_HOME',Path(__file__).resolve().parents[1])).resolve()
 print(prepare(root/'artifacts/huihui-six-thinking'))
