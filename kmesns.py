import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

def pcap(x,l1,l2,l3):
    pca=decomposition.PCA(n_components=2)
    pca.fit(x)
    X_new = pca.transform(x)
    p1 = plt.scatter(X_new[0:l1, 0], X_new[0:l1, 1], marker = 'x', color = 'red')
    p2 = plt.scatter(X_new[l1:l2, 0], X_new[l1:l2, 1], marker = '+', color = 'blue')
    p3 = plt.scatter(X_new[l2:l3, 0], X_new[l2:l3, 1], marker = 'o', color = 'yellow')
    plt.show()

def kmeans(x):
    z=np.zeros([6,4])
    z[3,:]=np.mean(x[0:50,:],axis = 0)#取每50个样本点的均值为初始值
    z[4,:]=np.mean(x[50:100,:],axis = 0)
    z[5,:]=np.mean(x[100:150,:],axis = 0)
    distance=np.zeros(3)
    while((np.allclose(z[0,:],z[3,:]) and np.allclose(z[1,:],z[4,:]) and np.allclose(z[2,:],z[5,:]))==False):
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
        w=np.vstack([x1[1:len(x1),:],x2[1:len(x2),:]])
        w=np.vstack([w,x3[1:len(x3),:]])
    pcap(w,len(x1)-1,len(x1)+len(x2)-2,150)
    print("各类样本数：",len(x1)-1,len(x2)-1,len(x3)-1)
    return w


iris = pd.read_csv('iris.data',header=None,sep=',')
iris1 = iris.iloc[0:150,0:4]
iris2 = np.mat(iris1)
x=kmeans(iris2)







