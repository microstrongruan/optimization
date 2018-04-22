import numpy as np
from functions import Penalty2, BoxThreeDimensional, PDQ
from optimizer import Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS
'''all input and output should be passed as numpy array with dtype np.float64
'''



for m in [3,5,10,15,20]:
    print("=================================================================================")
    print("===========================optimize for Box3D with m="+str(m)+"===========================")
    print("=================================================================================")
    object_f = BoxThreeDimensional(m=m)

    for opt in [Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS]:
        start_point = np.array([0,10,20],np.float64)
        print("-----optimize with "+opt.get_name()+"; start_point is "+str(start_point)+"-----")
        object_optimizer = opt(obj_f=object_f)
        object_optimizer.optimize(start_point=start_point,verbose=False)



