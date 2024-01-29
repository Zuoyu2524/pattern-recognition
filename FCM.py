# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:11:05 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

def define(x,U,z,a):#目标函数
    J=np.zeros(1)
    for j in range(3):
        for i in range(150):
            J=J+np.dot(U[i,j]**a,np.linalg.norm(x[i,0:4]-z[j,0:4])**2)
    return J

def belong(x,z,a):#隶属度矩阵更新
    U=np.zeros([150,3])
    for i in range(150):
        for j in range(3):
            for k in range(3):
                U[i,j]+=(np.linalg.norm(x[i,0:4]-z[j,0:4])/np.linalg.norm(z[k,0:4]-x[i,0:4]))**(2/(a-1))
            U[i,j]=1/U[i,j]
    return U

def renew(x,U,a):#聚类中心更新
    z1=np.zeros(1)
    z=np.zeros([3,4])
    for j in range(3):
        for i in range(150):
            z[j,0:4]=z[j,0:4]+np.dot((U[i,j]**a),x[i,0:4])
            z1=z1+U[i,j]**a
        z[j,0:4]=z[j,0:4]/z1
    return z

def pcap(x,l1,l2,l3):
    pca=decomposition.PCA(n_components=2)
    pca.fit(x)
    X_new = pca.transform(x)
    p1 = plt.scatter(X_new[0:l1, 0], X_new[0:l1, 1], marker = 'x', color = 'red')
    p2 = plt.scatter(X_new[l1:l2, 0], X_new[l1:l2, 1], marker = '+', color = 'blue')
    p3 = plt.scatter(X_new[l2:l3, 0], X_new[l2:l3, 1], marker = 'o', color = 'yellow')
    plt.show()

def FCM(x,point,a):
    x1=np.zeros([1,4])
    x2=np.zeros([1,4])
    x3=np.zeros([1,4])
    z=np.zeros([3,4])
    z=point
    J=np.zeros(1)
    U=belong(x,z,a)
    J=define(x,U,z,a)
    print(U)
    while (J<0.01):
        z=renew(x,U,a)
        U=belong(x,z,a)
        J=define(x,U,z,a)
    for i in range(150):
        l=np.where(U[i,:]==np.max(U[i,:]))
        if l==np.array([0]):
            x1=np.vstack([x1,x[i,:]])
        if l==np.array([1]):
            x2=np.vstack([x2,x[i,:]])
        if l==np.array([2]):
            x3=np.vstack([x3,x[i,:]])
    w=np.vstack([x1[1:len(x1),:],x2[1:len(x2),:]])
    w=np.vstack([w,x3[1:len(x3),:]])
    pcap(w,len(x1)-1,len(x1)+len(x2)-2,150)
    return {1:U,2:z}

def kmeans(x):
    z=np.zeros([6,4])
    z[3,:]=np.mean(x[0:50,:],axis = 0)#取每50个样本点的均值为初始值
    z[4,:]=np.mean(x[50:100,:],axis = 0)
    z[5,:]=np.mean(x[100:150,:],axis = 0)
    distance=np.zeros(3)
    while((np.allclose(z[0,:],z[3,:],rtol=1) and np.allclose(z[1,:],z[4,:],rtol=1) and np.allclose(z[2,:],z[5,:],rtol=1))==False):
        x1=np.zeros([1,4])
        x2=np.zeros([1,4])
        x3=np.zeros([1,4])
        z[0,:]=z[3,:]
        z[1,:]=z[4,:]
        z[2,:]=z[5,:]
        for i in range(150):
            distance[0]=np.linalg.norm(x[i,0:4]-z[0,0:4])
            distance[1]=np.linalg.norm(x[i,0:4]-z[1,0:4])
            distance[2]=np.linalg.norm(x[i,0:4]-z[2,0:4])
            l=np.where(distance==np.min(distance))
            if l==np.array([0]):
                x1=np.vstack([x1,x[i,:]])
            elif l==np.array([1]):
                x2=np.vstack([x2,x[i,:]])
            else:
                x3=np.vstack([x3,x[i,:]])
        z[3,:]=np.mean(x1[1:len(x1),:],axis = 0)
        z[4,:]=np.mean(x2[1:len(x2),:],axis = 0)
        z[5,:]=np.mean(x3[1:len(x3),:],axis = 0)
    return z

iris = pd.read_csv('iris.data',header=None,sep=',')
iris1 = iris.iloc[0:150,0:4]
iris2 = np.mat(iris1)
point1=kmeans(iris2)
point=point1[3:6,:]
x=FCM(iris2,point,1.5)




