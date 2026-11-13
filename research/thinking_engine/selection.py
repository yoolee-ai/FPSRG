"""Frozen six-axis development selection; no generation or fleet control."""
import collections, shutil
import common as c

def output_records(stage):return [c.read(p) for p in sorted((c.RUN/'stages'/stage/'outputs').glob('*.json'))]

def job_of(r):return r['job']

def select():
    if (c.RUN/'selection.json').exists():return
    from scoring import score
    groups=collections.defaultdict(list)
    for r in output_records('dev'):
        m=job_of(r)['metadata'];groups[m['axis'],m['layer']].append((m,score(m['case'],r)))
    all_options={};chosen={}
    for axis in c.AXES:
        options=[]
        for layer in c.LAYERS:
            obs=groups[axis,layer]
            if len(obs)!=60:raise RuntimeError('Incomplete_dev_candidate')
            means={}
            for kind in ['Q','B']:
                for alpha in [-1.,0.,1.]:
                    values=[v for m,v in obs if m['case']['kind']==kind and m['alpha']==alpha]
                    means[f'{kind}/{alpha}']=sum(50. if v is None else v for v in values)/len(values)
            valid=sum(v is not None for _,v in obs)/len(obs)
            gap=.5*(means['Q/1.0']-means['Q/-1.0'])+.5*(means['B/1.0']-means['B/-1.0'])
            path=c.RUN/f'candidates/{axis}_L{layer}/vector.pt'
            options.append(dict(axis=axis,layer=layer,separation=gap,valid_rate=valid,merit=gap-100*(1-valid),
                development_means=means,file=str(path.relative_to(c.RUN)),vector_sha=c.sha(path)))
        eligible=[x for x in options if x['valid_rate']>=.98]
        best=max(eligible or options,key=lambda x:(x['merit'],-x['layer']))
        dest=c.RUN/f'selected/{axis}.pt';dest.parent.mkdir(exist_ok=True);shutil.copy2(c.RUN/best['file'],dest)
        chosen[axis]=dict(best,file=str(dest.relative_to(c.RUN)),development_gate_passed=bool(eligible));all_options[axis]=options
    c.save(c.RUN/'selection_candidates.json',all_options);c.save(c.RUN/'selection.json',chosen)

