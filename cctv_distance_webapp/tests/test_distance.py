import numpy as np
from server import triangulate

def test_synthetic_triangulation():
    K=np.array([[800.,0,640],[0,800.,360],[0,0,1.]])
    P1=K@np.hstack([np.eye(3),np.zeros((3,1))])
    P2=K@np.hstack([np.eye(3),np.array([[1.],[0.],[0.]])])
    X=np.array([2.,1.,20.])
    a=P1@np.r_[X,1.]; b=P2@np.r_[X,1.]
    pa=(a[:2]/a[2]).tolist(); pb=(b[:2]/b[2]).tolist()
    est,e1,e2=triangulate(P1,P2,pa,pb)
    assert np.linalg.norm(est-X)<1e-6
    assert e1<1e-6 and e2<1e-6
