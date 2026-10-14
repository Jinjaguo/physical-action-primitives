"""Statistical and selection contract checks for the Query/Invoke summary."""
import numpy as np
from physical_action_primitives.query_invoke.summarize import interval, vectors


def main():
    # Pairing must remove shared parent difficulty exactly.
    baseline=np.arange(60,dtype=float).reshape(3,20)
    result=interval((baseline-2)-baseline)
    np.testing.assert_allclose(result['ci95'],[-2,-2])
    np.testing.assert_allclose(result['simultaneous_family6_interval'],[-2,-2])
    assert result['mean_delta']==-2
    # Preserve the source-block axis when averaging seed replicas.
    z={f'query__affine_{seed}_fixed_source':np.arange(20)+seed for seed in range(4)}
    z['query__affine_0_new_source']=np.full(20,9999.)
    np.testing.assert_allclose(vectors(z,'affine','query'),np.arange(20)+1.5)
    # Shared selector must keep deterministic first-index ties and use scaled six-field error.
    scale=np.array([.005,.005,.0872664626,.05,.05,1.])
    actual=np.array([[.005,0,0,0,0,0],[0,0,0,.05,0,0],[.010,0,0,0,0,0]])
    loss=((actual/scale)**2).sum(-1)
    np.testing.assert_allclose(loss,[1,1,4]);assert loss.argmin()==0
    # Bootstrap resamples complete paired blocks; equal inputs give exact zero intervals.
    np.testing.assert_allclose(interval(baseline-baseline)['ci95'],[0,0])
    print('Query/Invoke selection and source-block statistical contract checks PASS',flush=True)


if __name__=='__main__':main()
