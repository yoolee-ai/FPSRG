"""Caller-configurable output directory for native-thinking vector refinement."""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

SYSTEM = '你是一个人，根据场景事实回答。\n你是本次测评的答题者。以下陈述中的“我”均指你，不是正在向你提问的人。请判断你自己的倾向。'
AXES = list('ACNRMT')
LAYERS = [8, 16, 24, 32, 40, 48, 56, 60]
PACKAGE_ROOT = Path(os.environ.get('FPSRG_HOME', Path(__file__).resolve().parents[2])).expanduser().resolve()
MODEL = PACKAGE_ROOT / 'models/huihui-six'
_run = Path(os.environ.get('FPSRG_RESEARCH_RUN', 'outputs/thinking-refinement').replace('${FPSRG_HOME}', str(PACKAGE_ROOT))).expanduser()
RUN = _run.resolve() if _run.is_absolute() else (PACKAGE_ROOT / _run).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def check_stop():
    if (RUN / 'STOP').exists(): raise RuntimeError('Research run stopped')


def status(rank, phase, **kwargs):
    save(RUN / 'status' / (str(rank) + '.json'), dict(phase=phase, time=time.time(), **kwargs))
