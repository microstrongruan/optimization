import numpy as np
from functions import Penalty2, BoxThreeDimensional, PDQ, Exponential_fit
from optimizer import Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS
'''all input and output should be passed as numpy array with dtype np.float64
'''


for m in [3,5,10,15,20]:
    start_point = np.array([0, 10, 20], np.float64)
    print("=================================================================================")
    print("===========================optimize for Box3D with m="+str(m)+"===========================")
    print("=================================================================================")
    object_f = BoxThreeDimensional(m=m)

    for opt in [Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS]:
        print("-----optimize with "+opt.get_name()+"; start_point is "+str(start_point)+"-----")
        object_optimizer = opt(obj_f=object_f)
        object_optimizer.optimize(start_point=start_point,verbose=False)



for n in [2,4,6,8,10]:
    start_point = np.array([1]*n, np.float64)
    print("=================================================================================")
    print("==========================optimize for Penalty with n="+str(n)+"==========================")
    print("=================================================================================")
    object_f = Penalty2(n=n)

    for opt in [Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS]:
        print("-----optimize with "+opt.get_name()+"; start_point is "+str(start_point)+"-----")
        object_optimizer = opt(obj_f=object_f)
        object_optimizer.optimize(start_point=start_point,verbose=False)



print("==================================solution for p153==================================")
object_f = Exponential_fit()
start_point = np.array([2,3,3,6,1],np.float64)
object_optimizer=Basic_Newton(object_f)
for opt in [Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS]:
    print("-----optimize with "+opt.get_name()+"; start_point is "+str(start_point)+"-----")
    object_optimizer = opt(obj_f=object_f)
    object_optimizer.optimize(start_point=start_point,verbose=False)
