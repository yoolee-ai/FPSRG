"""Install an exact weight bundle from an internal store into FPSRG_HOME/models."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
 return h.hexdigest()

def install(root,name,source,verify_only=False):
 root=Path(root).resolve();source=Path(source).resolve();dest=root/'models'/name
 manifest=json.loads((root/'models'/f'{name}.files.json').read_text())
 for rel,entry in manifest.items():
  if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Invalid manifest path')
  src=source/rel;dst=dest/rel
  if not src.is_file() or src.stat().st_size!=entry['size'] or sha(src)!=entry['sha256']:
   raise ValueError('Source mismatch: '+rel)
  if verify_only:continue
  dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():
   if dst.stat().st_size==entry['size'] and sha(dst)==entry['sha256']:continue
   raise ValueError('Different existing destination: '+rel)
  fd,tmp=tempfile.mkstemp(dir=dst.parent,prefix='.weight-')
  os.close(fd)
  try:
   shutil.copyfile(src,tmp)
   if sha(tmp)!=entry['sha256']:raise ValueError('Copy checksum mismatch: '+rel)
   os.replace(tmp,dst)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 if name=='huihui-six' and not verify_only:
  notice=root/'models/Huihui-LICENSE.txt'
  target=dest/'Huihui-LICENSE.txt'
  if target.exists() and target.read_bytes()!=notice.read_bytes():
   raise ValueError('Existing model license differs')
  if not target.exists():shutil.copyfile(notice,target)
 print(f'{name}: {len(manifest)} files '+('verified' if verify_only else 'installed'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['qwen35-base','persononly-lora','huihui-six'],required=True)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--verify-only',action='store_true')
 p.add_argument('--root',type=Path,default=Path(os.environ.get('FPSRG_HOME',Path(__file__).resolve().parents[1])))
 a=p.parse_args();install(a.root,a.name,a.source,a.verify_only)
