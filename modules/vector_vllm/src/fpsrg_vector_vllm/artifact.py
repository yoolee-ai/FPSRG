"""Portable artifact loading, catalog matching and immutable cache identities."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def fingerprint(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class Artifact:
    def __init__(self, manifest):
        self.path = Path(manifest).expanduser().resolve()
        self.root = self.path.parent
        self.spec = json.loads(self.path.read_text())
        s = self.spec
        if s.get('schema_version') != 1 or s.get('backend') not in ('ten_off', 'huihui6_full'):
            raise ValueError('Unsupported artifact schema/backend')
        self.backend = s['backend']
        self.axes = s['axes']
        if not isinstance(self.axes, list) or len(set(self.axes)) != len(self.axes) or not self.axes:
            raise ValueError('Unique axes required')
        if not fingerprint(s['model']['fingerprint']):
            raise ValueError('model.fingerprint must be a SHA256 identity')
        self.model = self.resolve(s['model']['path'])
        self.generation = dict(system='你是一个人，根据场景事实回答。',
                               thinking=self.backend == 'huihui6_full', max_model_len=32768,
                               max_tokens=4096, **{})
        self.generation.update(s.get('generation', {}))
        if type(self.generation['thinking']) is not bool or self.generation['thinking'] != (self.backend == 'huihui6_full'):
            raise ValueError('Thinking mode must match the calibrated artifact')
        self.emotion = s.get('emotion')
        if self.emotion and (self.backend != 'huihui6_full' or self.emotion['modes'] != ['none','joy','anger']
                             or self.emotion['strength'] != 2.0):
            raise ValueError('Only the calibrated Huihui emotion overlay recipe is supported')
        catalog = json.loads(self.asset('catalog').read_text())
        self.nodes = catalog['nodes'] if isinstance(catalog, dict) else catalog
        self.by_node = {}
        for node in self.nodes:
            ident, persona = node['node_id'], node['persona_id']
            if type(ident) is not int or ident < 0 or ident in self.by_node or type(persona) is not int or persona < 0:
                raise ValueError('Invalid or duplicate node/persona identity')
            if set(node['scores']) != set(self.axes) or set(node['coefficients']) != set(self.axes):
                raise ValueError('Catalog axes mismatch')
            for key in ('scores', 'coefficients'):
                if any(type(v) not in (int,float) or not math.isfinite(v) for v in node[key].values()):
                    raise ValueError('Finite catalog values required')
            if any(not 0 <= v <= 100 for v in node['scores'].values()):
                raise ValueError('Scores must be in [0,100]')
            self.by_node[ident] = node
        if not self.nodes:
            raise ValueError('No eligible catalog nodes')
        if self.backend == 'ten_off':
            if self.axes != list('OCEANRTVMS') or not s.get('adapter') or self.emotion:
                raise ValueError('ten_off requires the original ten-axis LoRA artifact')
            self.asset('vectors')
        else:
            if self.axes != list('ACNRMT') or s.get('adapter') is not None:
                raise ValueError('Huihui full checkpoint must not load a LoRA')
            self.asset('bank')
        # Files are content-addressed; local deployment paths are intentionally not cache identity.
        self.identity = digest(dict(backend=self.backend, model=s['model']['fingerprint'],
            adapter=None if not s.get('adapter') else s['adapter']['commit_sha256'],
            assets={k: v['sha256'] for k,v in s['assets'].items()},
            layers=s.get('layer_by_trait'), generation=self.generation, emotion=self.emotion))

    def resolve(self, value):
        expanded = os.path.expandvars(str(value))
        if '$' in expanded:
            raise ValueError('Unresolved environment variable in artifact path')
        path = Path(expanded).expanduser()
        return (path if path.is_absolute() else self.root/path).resolve()

    def asset(self, name):
        row = self.spec['assets'][name]
        path = self.resolve(row['path'])
        if not fingerprint(row['sha256']) or sha256(path) != row['sha256']:
            raise ValueError('Asset SHA256 mismatch: '+name)
        return path

    def choose(self, node_id=None, targets=None):
        if node_id is not None:
            if targets is not None or type(node_id) is not int or node_id not in self.by_node:
                raise ValueError('Choose an eligible node_id OR targets')
            return dict(self.by_node[node_id])
        targets = {a:50.0 for a in self.axes} if targets is None else targets
        if not isinstance(targets, dict) or set(targets) != set(self.axes):
            raise ValueError('Exactly one target for every axis is required')
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 100 for v in targets.values()):
            raise ValueError('Targets must be finite scores in [0,100]')
        node = min(self.nodes, key=lambda n:(sum(abs(n['scores'][a]-targets[a]) for a in self.axes), n['node_id']))
        return dict(node, requested_scores=dict(targets), l1_error=sum(abs(node['scores'][a]-targets[a]) for a in self.axes))

    def route(self, node, emotion='none'):
        if emotion not in ('none', 'joy', 'anger'):
            raise ValueError('Unknown emotion')
        row = node['persona_id']
        if emotion != 'none':
            if not self.emotion:
                raise ValueError('This artifact has no emotion vectors')
            if not 0 <= row < self.emotion['base_rows']:
                raise ValueError('Invalid logical persona row')
            row += self.emotion['base_rows'] * self.emotion['modes'].index(emotion)
        return row

    def cache_salt(self, persona):
        return 'fpsrg-v1:'+digest(dict(identity=self.identity, persona=persona))

    def public(self):
        return dict(artifact_id=self.spec['artifact_id'], backend=self.backend, axes=self.axes,
                    nodes=self.nodes, generation=self.generation, emotion=self.emotion,
                    gate=self.spec.get('gate'), model_fingerprint=self.spec['model']['fingerprint'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('manifest')
    args=parser.parse_args(); art=Artifact(args.manifest)
    print(json.dumps(dict(artifact_id=art.spec['artifact_id'], backend=art.backend,
                         eligible_nodes=len(art.nodes), identity=art.identity),indent=2))
