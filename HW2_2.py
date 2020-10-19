import  numpy as np
import cmath
import os
import copy
def weight_update(new_weight,trainx):
    upadte_weight = new_weight+(0.5*(trainx-new_weight))
    return upadte_weight
def training(pattern,weight):
    pass
    weight_list=[weight]
    newweight = copy.deepcopy(weight)
    for i in range(2):
        for single_pattern in pattern:

            compete_result=np.dot(single_pattern,np.transpose(newweight))
            max_index=int(np.where(compete_result== np.max(compete_result, axis=0))[0][0])
            newweight[max_index]=weight_update(new_weight=newweight[max_index],trainx=single_pattern)
            storeweight=copy.deepcopy(newweight)
            weight_list.append(storeweight)

            pass
    return weight_list


if __name__ == '__main__':
    pattern=np.array([[-1,0],[0,1],[1/cmath.sqrt(2),1/cmath.sqrt(2)]])
    weight=np.array([[0,-1],[-2/cmath.sqrt(5),1/cmath.sqrt(5)],[-1/cmath.sqrt(5),2/cmath.sqrt(5)]])
    print(weight)
    weight_list=training(pattern=pattern,weight=weight)
    print(weight_list)
    for index,single_weight in enumerate(weight_list):
        print('update times:{}'.format(index))
        print(single_weight)
