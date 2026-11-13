"""Paper coverage: same-node L-infinity <=20 and L1 <= supplied limit."""
import numpy as np

def measure(scores,targets,gate,limit):
    n=len(targets);out={};ev={}
    for name,mask in [('raw',np.ones(len(scores),bool)),('gate98',gate)]:
        ids=np.flatnonzero(mask);ss=scores[ids]
        nn=np.zeros(n,np.int64);nl1=np.zeros(n);ninf=np.zeros(n)
        fi=np.full(n,-1,np.int64);fl1=np.full(n,np.inf);finf=np.full(n,np.inf)
        for start in range(0,n,24):
            ts=targets[start:start+24];d=np.zeros((len(ts),len(ss)));inf=np.zeros_like(d)
            for axis in range(scores.shape[1]):
                x=np.abs(ts[:,None,axis]-ss[None,:,axis]);d+=x;np.maximum(inf,x,out=inf)
            rr=np.arange(len(ts));ix=d.argmin(1);sl=slice(start,start+len(ts))
            nn[sl]=ids[ix];nl1[sl]=d[rr,ix];ninf[sl]=inf[rr,ix]
            d[inf>20]=np.inf;ix=d.argmin(1);v=d[rr,ix];ok=np.isfinite(v)
            fi[sl]=np.where(ok,ids[ix],-1);fl1[sl]=v;finf[sl]=np.where(ok,inf[rr,ix],np.inf)
        covered=fl1<=limit;returned=(nl1<=limit)&(ninf<=20)
        assert np.all(returned<=covered)
        out[name]=dict(nodes=int(mask.sum()),targets=n,covered=int(covered.sum()),coverage_percent=float(covered.mean()*100),ordinary_L1_returned_pass=int(returned.sum()),ordinary_L1_returned_percent=float(returned.mean()*100),mean_nearest_L1=float(nl1.mean()),mean_nearest_Linf=float(ninf.mean()))
        ev.update({name+'_'+k:v for k,v in dict(nearest_node=nn,nearest_L1=nl1,nearest_Linf=ninf,feasible_node=fi,feasible_L1=fl1,feasible_Linf=finf).items()})
    assert np.all((ev['gate98_feasible_L1']<=limit)<=(ev['raw_feasible_L1']<=limit))
    return out,ev

