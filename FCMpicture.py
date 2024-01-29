# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:57:00 2020

@author: Lenovo
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

def define(r,l,x,U,z,a):#目标函数
    J=np.zeros(1)
    for j in range(2):
        for i in range(int(r*l)):
            J=J+np.dot(U[i,j]**a,(x[i]-z[j])**2)
    return J

def belong(r,l,x,z,a):#隶属度矩阵更新
    U=np.zeros([int(r*l),2])
    for i in range(int(r*l)):
        for j in range(2):
            for k in range(2):
                U[i,j]+=((x[i]-z[j])/(z[k]-x[i]))**(2/(a-1))
            U[i,j]=1/U[i,j]
    return U

def renew(r,l,x,U,a):#聚类中心更新
    z1=np.zeros(1)
    z=np.zeros([2,1])
    for j in range(2):
        for i in range(int(r*l)):
            z[j]=z[j]+np.dot((U[i,j]**a),x[i])
            z1=z1+U[i,j]**a
        z[j]=z[j]/z1
        z1=np.array([0])
    return z

def kmeans(x,p):
    z=np.zeros([4,1])
    z[2]=np.mean(x[0:int(p/2)],axis = 0)
    z[3]=np.mean(x[int(p/2):int(p)],axis = 0)
    distance=np.zeros(2)
    while((np.allclose(z[0,0],z[2,0],rtol=1e-1) and np.allclose(z[1,0],z[3,0],rtol=1e-1))==False):
        x1=np.zeros([1,1])
        x2=np.zeros([1,1])
        z[0,0]=z[2,0]
        z[1,0]=z[3,0]
        for i in range(p):
            distance[0]=np.linalg.norm(x[i,0]-z[0,0])
            distance[1]=np.linalg.norm(x[i,0]-z[1,0])
            l=np.where(distance==np.min(distance))
            if l==np.array([0]):
                x1=np.vstack([x1,x[i,:]])
            else:
                x2=np.vstack([x2,x[i,:]])
        z[2,0]=np.mean(x1[1:len(x1),0],axis = 0)
        z[3,0]=np.mean(x2[1:len(x2),0],axis = 0)
    return z

def FCM(x,point,a,r,l):
    times=0
    x1=np.zeros([int(r*l),1])
    z=np.zeros([2,1])
    z=point
    J=np.zeros(2)
    U=belong(r,l,x,z,a)
    J[0]=define(r,l,x,U,z,a)
    while (np.abs(J[0]-J[1])>0.001):
        J[1]=J[0]
        z=renew(r,l,x,U,a)
        U=belong(r,l,x,z,a)
        J[0]=define(r,l,x,U,z,a)
        times=times+1
        print(times,J[0],'    ',J[1])
    for i in range(int(r*l)):
        l=np.where(U[i,:]==np.max(U[i,:]))
        if l==np.array([0]):
            x1[i]=0
        else:
            x1[i]=255
    return {1:U,2:x1}

def divide(img,a):
    rows, cols = img.shape[:2]
    pixel_count = rows * cols
    img_array = img.reshape(pixel_count,1)
    z=kmeans(img_array,pixel_count)
    z1=z[2:4]
    x=FCM(img_array,z1,2,rows,cols)
    new_img = x.get(2).reshape(rows, cols)
    cv2.imshow("result", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

img1=cv2.imread("E:/1.bmp",cv2.IMREAD_GRAYSCALE)
img2=cv2.imread("E:/2.bmp",cv2.IMREAD_GRAYSCALE)
img3=cv2.imread("E:/3.bmp",cv2.IMREAD_GRAYSCALE)
y=1
if y>0:
    y=divide(img1,2)

