# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 22:53:08 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

def pcap(x,l1,l2):
    pca=decomposition.PCA(n_components=2)
    pca.fit(x)
    X_new = pca.transform(x)
    p1 = plt.scatter(X_new[0:l1, 0], X_new[0:l1, 1], marker = 'x', color = 'red')
    p2 = plt.scatter(X_new[l1:l2, 0], X_new[l1:l2, 1], marker = '+', color = 'blue')
    plt.show()

def kmeans(x):
    z=np.zeros([4,60])
    z[2,:]=np.mean(x[0:104,:],axis = 0)#取每50个样本点的均值为初始值
    z[3,:]=np.mean(x[104:208,:],axis = 0)
    distance=np.zeros(2)
    while((np.allclose(z[0,:],z[2,:]) and np.allclose(z[1,:],z[3,:]))==False):
        x1=np.zeros([1,60])
        x2=np.zeros([1,60])
        z[0,:]=z[2,:]
        z[1,:]=z[3,:]
        for i in range(208):
            distance[0]=np.linalg.norm(x[i,0:60]-z[0,0:60])
            distance[1]=np.linalg.norm(x[i,0:60]-z[1,0:60])
            l=np.where(distance==np.min(distance))
            if l==np.array([0]):
                x1=np.vstack([x1,x[i,:]])
            else:
                x2=np.vstack([x2,x[i,:]])
        z[2,:]=np.mean(x1[1:len(x1),:],axis = 0)
        z[3,:]=np.mean(x2[1:len(x2),:],axis = 0)
        w=np.vstack([x1[1:len(x1),:],x2[1:len(x2),:]])
    pcap(w,len(x1)-1,208)
    print("各类样本数：",len(x1)-1,len(x2)-1)
    return w

sonar = pd.read_csv('sonar.all-data',header=None,sep=',')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.mat(sonar1)
x=kmeans(sonar2)
