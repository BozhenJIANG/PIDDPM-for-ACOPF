'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2026-03-18 20:15:04
LastEditors: JIANG Bozhen
LastEditTime: 2026-03-20 19:56:03
'''
import numpy as np
from matplotlib import pyplot as plt

X_con_train_ = []
with open("./Dataset/X_con_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_con_train_.append([float(_i) for _i in _data])

X_in_train_ = []
with open("./Dataset/X_in_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_in_train_.append([float(_i) for _i in _data])

X_con_test_ = []
with open("./Dataset/X_con_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_con_test_.append([float(_i) for _i in _data])

X_in_test_ = []
with open("./Dataset/X_in_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_in_test_.append([float(_i) for _i in _data])

X_in_train = np.array(X_in_train_)
X_in_train[:,:54] = X_in_train[:,:54]/100
X_con_train = np.array(X_con_train_)/100

X_in_test = np.array(X_in_test_)
X_in_test[:,:54] = X_in_test[:,:54]/100
X_con_test = np.array(X_con_test_)/100



from matplotlib import pyplot as plt
fig = plt.figure(figsize=(5,2))
plt.hist(X_con_train[:,0]*100,density=True,label="Training Set")
plt.hist(X_con_test[:,0]*100,density=True,label="Testing Set")
plt.xlabel("Load/MW")
plt.ylabel("Probability Distribution \n Density")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
fig.savefig("./distribution.svg",dpi=300)


