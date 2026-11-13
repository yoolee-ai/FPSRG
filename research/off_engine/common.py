"""Portable research paths; no cluster scheduler or process termination."""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


CONFIG_PATH = Path(os.environ.get('FPSRG_RESEARCH_CONFIG', Path(__file__).with_name('config.example.json'))).resolve()
CONFIG = read(CONFIG_PATH)
PACKAGE_ROOT = Path(os.environ.get('FPSRG_HOME', Path(__file__).resolve().parents[2])).expanduser().resolve()


def configured_path(key):
    value = Path(CONFIG[key].replace('${FPSRG_HOME}', str(PACKAGE_ROOT))).expanduser()
    return value if value.is_absolute() else (CONFIG_PATH.parent / value).resolve()


RUN = configured_path('run')
ROOT_RUN = configured_path('root_run')
BASE = MODEL = configured_path('model')
PARENT = configured_path('adapter_checkpoint')
AXES = tuple(CONFIG['axes'])


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


def status(who, phase, **kwargs):
    save(RUN / 'status' / (str(who) + '.json'), dict(phase=phase, time=time.time(), **kwargs))


class CandidateAuditFailed(RuntimeError):
    pass


def budget():
    if any((root / 'STOP').exists() or (root / 'stop_requested.json').exists() for root in (RUN, ROOT_RUN)):
        raise RuntimeError('Research run stopped')
