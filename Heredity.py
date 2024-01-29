# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:23:15 2020

@author: Lenovo
"""

import numpy as np
import math as m
import random as rand
import matplotlib.pyplot as plt

#解码函数
def decode(binary,a,b,l):
    bina = ''.join(str(i) for i in binary)
    x=a+int(bina,2)*((b-a)/(2**l-1))
    return x
    
#初始化20个种群样本，将其存到numpy数组中
def code(a,b,lmax):
    num = np.random.randint(2,size = (1,int(lmax)))
    n=0
    while n<19:
        pop = np.random.randint(2,size = (1,int(lmax)))
        num=np.vstack([num,pop])
        n=n+1
    return num

#适应度评价
def peval(choice,x):
    if choice==1:
        val=x[0]**2+x[1]**2
    elif choice==2:
        val=abs(x[0])+abs(x[1])+abs(x[1])*abs(x[0])
    elif choice==3:
        val=2*x[0]**2+x[1]**2+2*x[0]*x[1]
    elif choice==4:
        val=x[0]**2+x[1]**2-10*(np.cos(2*m.pi*x[0])+np.cos(2*m.pi*x[1]))+20
    else:
        val=20*(1-np.exp(-0.2*np.sqrt(1/2*(x[0]**2+x[1]**2))))+np.exp(1)-np.exp(1/2*(np.cos(2*m.pi*x[0])+np.cos(2*m.pi*x[1])))
    return val

#轮盘赌选择
def selection(binary,choice,l,a,b):
    F=np.array([0])
    val_array=np.array([])
    pk=np.array([])
    qk=np.array([])
    for i in range(20):
        x1=decode(binary[i,0:l//2],a,b,l//2)
        x2=decode(binary[i,l//2:l],a,b,l//2)
        val_array=np.append(val_array,peval(choice,np.array([[x1],[x2]])))
        F=F+1/peval(choice,np.array([[x1],[x2]]))
    for i in range(20):
        pk=np.append(pk,1/val_array[i]/F)
    for i in range(21):
        p=0
        for j in range(i):
            p=p+pk[j]
        qk=np.append(qk,p)
    qk=np.delete(qk,0)
    p=rand.random()
    for k in range(20):
            if p<qk[k]:
                newnum=binary[k]
                break
    for i in range(19):
        p=rand.random()
        for k in range(20):
            if p<qk[k]:
                newnum=np.vstack([newnum,binary[k]])
                break
    return {1:newnum,2:val_array}

#交叉
def jiaocha(newnum,l,pc):
    for i in range(10):
        p=rand.random()
        while(p<pc):
            place=np.random.randint(1,l-1,1)
            temp=newnum[i*2,int(place):l]
            temp1=newnum[i*2,0:int(place)]
            newnum[i*2]=np.hstack([temp1,newnum[i*2+1,int(place):l]])
            temp2=newnum[i*2+1,0:int(place)]
            newnum[i*2+1]=np.hstack([temp2,temp])
            p=rand.random()
    return newnum

#变异
def bianyi(newnum,l,pc):
    for i in range(20):
        for j in range(l):
            p=rand.random()
            if p<pc:
                newnum[i,j]=abs(newnum[i,j]-np.array(1))
    return newnum

def main(a,b,choice):
    lmax=2*(int(m.log((b-a)*1e4+1,2))+1)
    num=code(a,b,lmax)
    num1=selection(num,choice,lmax,a,b)
    min_val=np.min(num1.get(2))
    mval=min_val
    for i in range(200):
        num2=jiaocha(num1.get(1),lmax,0.1)
        num3=bianyi(num2,lmax,0.05)
        num1=selection(num3,choice,lmax,a,b)
        min_val=np.min(num1.get(2))
        mval=np.hstack([mval,min_val])
    return mval

t=range(201)#默认迭代201次
result1=main(-100,100,1)
print('函数1经过迭代得到的最优值为：',np.min(result1))
plt.figure(1)
plt.plot(t,result1,'-')
plt.show()

result2=main(-10,10,2)
print('函数2经过迭代得到的最优值为：',np.min(result2))
plt.figure(2)
plt.plot(t,result2,'-')
plt.show()

result3=main(-100,100,3)
print('函数3经过迭代得到的最优值为：',np.min(result3))
plt.figure(3)
plt.plot(t,result3,'-')
plt.show()

result4=main(-5.12,5.12,4)
print('函数4经过迭代得到的最优值为：',np.min(result4))
plt.figure(4)
plt.plot(t,result4,'-')
plt.show()

result5=main(-32,32,5)
print('函数5经过迭代得到的最优值为：',np.min(result5))
plt.figure(5)
plt.plot(t,result5,'-')
plt.show()