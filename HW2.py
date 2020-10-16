import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import os
def weight_update(weight,trainy,learningrate,trainx):
    upadte_weight = weight + (trainy * learningrate *trainx)
    return upadte_weight
def training(trainx,trainy,weight,learningrate,epoch):
    weight_list=[weight]

    passnum=0
    for i in range(epoch):
        if i==0:
            newtrainx=trainx
            newtrainy=trainy
        else:
            per = np.random.permutation(trainx.shape[0])
            # 打亂後的行號
            newtrainx = trainx[per, :]
            # 獲取打亂後的訓練資料
            newtrainy = trainy[per]

        for single_trainx,single_trainy in zip(newtrainx,newtrainy):
            if (  sign(np.dot(weight,single_trainx) )!=sign(single_trainy)):
                print(weight_list[-1])
                weight_list.append(
                    weight_update(weight=weight_list[-1],trainy=single_trainy,learningrate=learningrate,trainx=single_trainx)   )
                print('pass{}'.format(passnum))
                # os.system('pause')
                break
            passnum+=1

        if passnum==7:
            break

        passnum=0
    return weight_list



if __name__ == '__main__':
    trainx=np.array([[0.08,0.72,1],[0.26,0.58,1],[0.45,0.15,1],[0.60,0.30,1],[0.1,1,1],[0.35,0.95,1],[0.70,0.65,1],[0.92,0.45,1]])
    trainy=np.array([1,1,1,1,-1,-1,-1,-1])
    x=trainx[:,0]
    y=trainx[:,1]
    weight=np.array([1,-1,0.2])


    weight_list=training(trainx=trainx,trainy=trainy,weight=weight,learningrate=0.15,epoch=35)

    f = lambda x: (weight[0]/-weight[1])*x+weight[2]
    graphx = np.array([-100, 100])
    plt.plot(graphx, f(graphx), c="red", label='line1')


    # colors = "bgrcmykw"
    for index,single_weight in  enumerate(weight_list):
        weight=single_weight
        print(weight)
        if index != len(weight_list) - 1:
            pass
            # plt.plot(graphx, f(graphx), c="red", label='line' + str(index))
        else:
            plt.plot(graphx,f(graphx), c="orange", label='line'+str(index))


    plt.scatter(x, y, alpha=0.6, label = 'data')
    plt.xlim(-1.2, 1.2)
    plt.ylim(0, 1.2)
    plt.legend()
    plt.show()