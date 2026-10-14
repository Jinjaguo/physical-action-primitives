"""Core geometry and configuration-validation tests for the organization audit."""
import numpy as np
from physical_action_primitives.organization.core import group_distances,physical_distance,witnessed,validate_config,response12

a=np.zeros((2,3,12));b=a.copy()
b[...,0]=.005
assert not witnessed(a,b).any(), 'strict equality must not separate'
b[:,0:2,0]=.0051
assert witnessed(a,b).all(), 'two branches must witness'
b=a.copy();b[:,0,0]=.01
assert not witnessed(a,b).any(), 'one branch insufficient'
a=np.zeros((1,12));b=a.copy();a[0,5]=np.pi;b[0,5]=-np.pi
assert group_distances(a,b)[0,1]<1e-7, 'same rotations must agree'
assert physical_distance(a,a)[0]<1e-7
trace=np.zeros((2,1,13));trace[...,3]=1;trace[-1,0,0]=.01
assert np.isclose(response12(trace,np.array([np.pi]))[0,0],-.01,atol=1e-7)
c=dict(source_xi_m=[[0,1]],audit_xi_m=[[0,1]],
       batches=[dict(seed=i,split=s) for i,s in enumerate(['train','development','query','gallery'])])
try: validate_config(c)
except ValueError: pass
else: raise AssertionError('source/audit overlap accepted')
print('ORGANIZATION CORE TESTS PASS')
