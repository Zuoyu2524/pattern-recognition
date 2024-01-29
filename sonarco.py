# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:39:40 2020

@author: Lenovo
"""
import numpy as np
import pandas as pd
from sklearn import cross_validation
sonar = pd.read_csv('sonar.all-data',header=None,sep=',')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.mat(sonar1)
Accuracy = np.zeros((60,3))
accuracy_ = np.zeros(10)
p=0.5

def Fisher(X1,X2,p):
    m1=(np.mean(X1,axis = 0))
    m2=(np.mean(X2,axis = 0))
    m1 = m1.reshape(60,1)   
    m2 = m2.reshape(60,1)
    S1 = np.zeros((60,60))             
    S2 = np.zeros((60,60))                                
    for i in range(0,int(96*(1-p))):
        S1+=(X1[i].reshape(60,1)-m1).dot((X1[i].reshape(60,1)-m1).reshape(1,60))
    for i in range(0,int(110*(1-p))):
        S2+=(X2[i].reshape(60,1)-m2).dot((X2[i].reshape(60,1)-m2).reshape(1,60))
    Sw=S1 + S2
    w=np.linalg.inv(Sw).dot(m1 - m2)
    W0=(np.dot(w.reshape(1,60),m1)+np.dot(w.reshape(1,60),m2))/2
    return {1:w,2:W0}

def Classify(X,W,W0):
    y=(W.T).dot(X)-W0
    return y

for i in range(0,60):
   train1,test1 = cross_validation.train_test_split(sonar2[0:96,:],test_size =p)
   train2,test2 = cross_validation.train_test_split(sonar2[97:208,:],test_size =p)
   test1= np.insert(test1,60, values=1, axis=1)
   test2= np.insert(test2,60, values=2, axis=1)
   test=np.vstack((test1,test2))
   test=np.random.permutation(test)
   m=Fisher(train1[:,0:60],train2[:,0:60],p)
   acr=0
   acr1=0
   acr2=0
   for j in range(0,int(96*p+110*p)):
       if Classify(test[j,0:60],m.get(1),m.get(2))<0:
           if test[j,60]==2:
               acr=acr+1
               acr2=acr2+1
       else:
           if test[j,60]==1:
               acr=acr+1
               acr1=acr1+1
   Accuracy[i,0]+=acr/(p*96+p*110)
   Accuracy[i,1]+=acr1/(p*96)
   Accuracy[i,2]+=acr2/(p*110)
print("Accuracy:%.3f %.3f %.3f"%(np.sum(Accuracy[:,0])/60,np.sum(Accuracy[:,1])/60,np.sum(Accuracy[:,2])/60))
    
import matplotlib.pyplot as plt

x = np.arange(1,61,1)
plt.xlabel('times')
plt.ylabel('Accuracy')
plt.ylim((0,1))           
plt.plot(x,Accuracy[:,0],'b')

