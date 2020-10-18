import numpy as np
import matplotlib.pyplot as plt
import copy
import os
def weight_update(weight,trainx,trainy):
    upadte_weight = weight+0.5*(np.multiply(trainy,trainx))
    return upadte_weight

if __name__ == '__main__':
    pass
    x=np.array([[2,-0.5,1],[2,0.5,2]])
    y=np.array([[3,0.5,1],[3,-0.5,2]])
    weight=np.array([1,-2,0.5])
    update_list=[weight]
    for singlex,singley in zip(x,y):
        update_list.append(weight_update(weight=update_list[-1],trainx=singlex,trainy=singley))

    for index,single_weight in enumerate(update_list):
        print('cycle:{}\n{}'.format(index,single_weight))