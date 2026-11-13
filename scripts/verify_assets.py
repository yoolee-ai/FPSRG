"""Check released assets without loading a model or touching a GPU."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from prepare_assets import prepare

def main():
 root=Path(os.environ.get('FPSRG_HOME',Path(__file__).resolve().parents[1])).resolve()
 prepare(root/'artifacts/huihui-six-thinking')
 for directory in sorted((root/'artifacts').iterdir()):
  if not (directory/'manifest.json').exists():continue
  spec=json.loads((directory/'manifest.json').read_text())
  for key,entry in spec['assets'].items():
   path=directory/entry['path']
   if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError(f'{directory.name}/{key} checksum mismatch')
  cat=json.loads((directory/spec['assets']['catalog']['path']).read_text())
  assert len(cat['nodes'])==spec['qualified_nodes']
  assert len({n['node_id'] for n in cat['nodes']})==len(cat['nodes'])
  for n in cat['nodes']:
   assert set(n['scores'])==set(spec['axes'])
   assert all(np.isfinite(x) and 0<=x<=100 for x in n['scores'].values())
  print(f"{directory.name}: {len(cat['nodes'])} nodes; assets verified")
if __name__=='__main__':main()
