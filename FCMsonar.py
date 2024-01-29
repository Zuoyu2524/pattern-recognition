# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:31:14 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

def define(x,U,z,a):#目标函数
    J=np.zeros(1)
    for j in range(2):
        for i in range(208):
            J=J+np.dot(U[i,j]**a,np.linalg.norm(x[i,0:60]-z[j,0:60])**2)
    return J

def belong(x,z,a):#隶属度矩阵更新
    U=np.zeros([208,2])
    for i in range(208):
        for j in range(2):
            for k in range(2):
                U[i,j]+=(np.linalg.norm(x[i,0:60]-z[j,0:60])/np.linalg.norm(z[k,0:60]-x[i,0:60]))**(2/(a-1))
            U[i,j]=1/U[i,j]
    return U

def renew(x,U,a):#聚类中心更新
    z1=np.zeros(1)
    z=np.zeros([2,60])
    for j in range(2):
        for i in range(208):
            z[j,0:60]=z[j,0:60]+np.dot((U[i,j]**a),x[i,0:60])
            z1=z1+U[i,j]**a
        z[j,0:60]=z[j,0:60]/z1
    return z

def pcap(x,l1,l2):
    pca=decomposition.PCA(n_components=2)
    pca.fit(x)
    X_new = pca.transform(x)
    p1 = plt.scatter(X_new[0:l1, 0], X_new[0:l1, 1], marker = 'x', color = 'red')
    p2 = plt.scatter(X_new[l1:l2, 0], X_new[l1:l2, 1], marker = '+', color = 'blue')
    plt.show()

def FCM(x,point,a):
    x1=np.zeros([1,60])
    x2=np.zeros([1,60])
    z=np.zeros([2,60])
    z=point
    J=np.zeros(1)
    U=belong(x,z,a)
    J=define(x,U,z,a)
    while (J<0.001):
        z=renew(x,U,a)
        U=belong(x,z,a)
        J=define(x,U,z,a)
    for i in range(208):
        l=np.where(U[i,:]==np.max(U[i,:]))
        if l==np.array([0]):
            x1=np.vstack([x1,x[i,:]])
        if l==np.array([1]):
            x2=np.vstack([x2,x[i,:]])
    w=np.vstack([x1[1:len(x1),:],x2[1:len(x2),:]])
    pcap(w,len(x1)-1,208)
    return {1:U,2:z}

def kmeans(x):
    z=np.zeros([4,60])
    z[2,:]=np.mean(x[0:104,:],axis = 0)#取每50个样本点的均值为初始值
    z[3,:]=np.mean(x[104:208,:],axis = 0)
    distance=np.zeros(2)
    while((np.allclose(z[0,:],z[2,:],rtol=1) and np.allclose(z[1,:],z[3,:],rtol=1))==False):
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
    return z

sonar = pd.read_csv('sonar.all-data',header=None,sep=',')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.mat(sonar1)
point1=kmeans(sonar2)
point=point1[2:4,:]
x=FCM(sonar2,point,2)