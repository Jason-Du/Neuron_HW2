import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import copy
import os
def weight_update(weight,trainy,learningrate,trainx):
    upadte_weight = weight + (trainy * learningrate *trainx)
    return upadte_weight
def weight_update2(weight,trainy,learningrate,trainx):
    upadte_weight= weight + (trainy * learningrate *trainx)
    upadte_weight[1]=copy.deepcopy(weight[1])
    return upadte_weight
def training(trainx,trainy,weight,learningrate,epoch):
    weight_list=[weight]
    passnum=0
    for i in range(epoch):
        passnum = 0
        per = np.random.permutation(trainx.shape[0])
        # 打亂後的行號
        newtrainx = trainx[per, :]
        # 獲取打亂後的訓練資料
        newtrainy = trainy[per]

        for single_trainx,single_trainy in zip(newtrainx,newtrainy):
            if (  sign(np.dot(weight_list[-1],single_trainx) )!=sign(single_trainy)):
                weight_list.append(
                weight_update(weight=weight_list[-1],trainy=single_trainy,learningrate=learningrate,trainx=single_trainx)   )

                break
            else:
                passnum+=1
        if passnum==8:
            print(i + 1)
            # print('jumppass{}'.format(passnum))
            break

        passnum = 0
    return weight_list
def training2(trainx,trainy,weight,learningrate,epoch):
    weight_list=[weight]
    for i in range(epoch):
        passnum = 0
        per = np.random.permutation(trainx.shape[0])
        # 打亂後的行號
        newtrainx = trainx[per, :]
        # 獲取打亂後的訓練資料
        newtrainy = trainy[per]

        for single_trainx,single_trainy in zip(newtrainx,newtrainy):
            # copy_single_trainx = copy.deepcopy(single_trainx)
            # copy_weight=copy.deepcopy(weight_list[-1])
            # del copy_single_trainx[1]
            # del copy_weight[1]
            copy_single_trainx = np.delete(single_trainx, 1)
            copy_weight = np.delete(weight_list[-1], 1)
            if (  sign(np.dot(copy_weight,copy_single_trainx) )!=sign(single_trainy)):
                weight_list.append(
                weight_update2(weight=weight_list[-1],trainy=single_trainy,learningrate=learningrate,trainx=single_trainx)   )
                break
        if passnum==8:
            print(i + 1)
            # print('jumppass{}'.format(passnum))
            break
    return weight_list
def find_best_line(weight,trainx,trainy):

    passnum_list=[]
    for single_weight in weight:
        passnum = 0
        for single_trainx, single_trainy in zip(trainx, trainy):
            if (  sign(np.dot(single_trainx,single_weight) )==sign(single_trainy)):
                passnum=passnum+1
        passnum_list.append(passnum)
    return passnum_list.index(max(passnum_list))

if __name__ == '__main__':
    trainx=np.array([[0.08,0.72,1],[0.26,0.58,1],[0.45,0.15,1],[0.60,0.30,1],[0.1,1,1],[0.35,0.95,1],[0.70,0.65,1],[0.92,0.45,1]])
    trainy=np.array([1,1,1,1,-1,-1,-1,-1])
    x=trainx[:,0]
    y=trainx[:,1]
    weight=np.array([1,-1,0.2])
    weight_list=training(trainx=trainx,trainy=trainy,weight=weight,learningrate=0.8,epoch=30)
    weight_list2=training2(trainx=trainx,trainy=trainy,weight=weight,learningrate=0.8,epoch=30)
    max_index=find_best_line(weight=weight_list2,trainx=trainx,trainy=trainy)
    print(max_index)
    f = lambda x: (weight[0]/-weight[1])*x-(weight[2]/weight[1])
    print(len(weight_list))
    print(len(weight_list2))
    plt.figure(1)
    graphx = np.array([-100, 100])
    plt.plot(graphx, f(graphx), c="red", label='Initial')
    weight = weight_list[1]
    plt.plot(graphx, f(graphx), c="green", label='line2')
    weight=weight_list[-1]
    plt.plot(graphx,f(graphx), c="orange", label='lastline')

    plt.legend()
    plt.scatter(x, y, alpha=0.6, label = 'data',s=10, marker='o')
    plt.xlim(-0.5, 1.2)
    plt.ylim(-0.5, 1.2)
    plt.figure(2)
    for weight_index,single_weight in enumerate(weight_list2):
        pass
        weight=single_weight
        if weight_index == 0:
            plt.plot(graphx, f(graphx), c="red", label='Initial')
            continue
        elif weight_index == max_index:
            plt.plot(graphx, f(graphx), c="orange", label='Line{}best'.format(weight_index+1))
            continue
        else:
            plt.plot(graphx, f(graphx), c="green")


    plt.legend()
    plt.scatter(x, y, alpha=0.6, label = 'data',s=10, marker='o')
    plt.xlim(-0.5, 1.2)
    plt.ylim(-0.5, 1.2)
    plt.show()
