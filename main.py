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
    object_optimizer = Basic_Newton
    optimizer = object_optimizer(f=object_f.calculate_value, g=object_f.calculate_derivative, G=object_f.calculate_hessian)

    for opt in [Basic_Newton,Zuni_Newton,LM_Newton,SR1,DFP,BFGS]:
        start_point = np.array([4,4,2],np.float64)
        print("-----optimize with "+opt.get_name()+"; start_point is "+str(start_point)+"-----")
        object_optimizer = opt(object_f.calculate_value, object_f.calculate_derivative, object_f.calculate_hessian)
        object_optimizer.optimize(start_point=start_point,verbose=False)



