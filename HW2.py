import numpy as np
import matplotlib.pyplot as plt
import os
trainx=np.array([[0.08,0.72,1],[0.26,0.58,1],[0.45,0.15,1],[0.60,0.30,1],[0.1,1,1],[0.35,0.95,1],[0.70,0.65,1],[0.92,0.45,1]])
trainy=np.array([1,1,1,1,-1,-1,-1,-1])
weight=np.array([1,-1,0.2])
learning_rate=0.4
# weight=weight+()





f = lambda x: (weight[0]/-weight[1])*x+weight[2]
x=trainx[:,0]
y=trainx[:,1]
graphx = np.array([0,100])
plt.scatter(x, y, alpha=0.6, label = 'data')
plt.plot(graphx,f(graphx), c="orange", label="LINE1")

# for update_times in range(8):
#     for error_index in range(8):
#         weight=weight+(   trainy[error_index]*
#                           learning_rate*
#                           (   np.dot(  trainx[error_index],weight )  )
#                     )
#         print(error_index)
#         if error_index==7:
#             plt.plot(graphx,f(graphx), c="red")

error_index=0
weight=weight+trainy[error_index]*learning_rate*(np.dot(trainx[error_index],weight))
plt.plot(graphx,f(graphx), c="red", label="LINE2")
error_index=1
weight=weight+trainy[error_index]*learning_rate*(np.dot(trainx[error_index],weight))
plt.plot(graphx,f(graphx), c="yellow", label="LINE2")
error_index=7
weight=weight+trainy[error_index]*learning_rate*(np.dot(trainx[error_index],weight))
plt.plot(graphx,f(graphx), c="blue", label="LINE2")
error_index=6
weight=weight+trainy[error_index]*learning_rate*(np.dot(trainx[error_index],weight))
plt.plot(graphx,f(graphx), c="olive", label="LINE2")

plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.legend()
plt.show()