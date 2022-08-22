import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.cm as cm

class Kmeans:
    def __init__(self,fileName,K,seed):
        random.seed(seed)
        self.seed=seed
        self.K=K
        file=open("data/%s.txt"%fileName,"r") 
        self.data=[]
        for string in file.readlines():
            temp=[float(s) for s in string[:-1].split(',')]
            self.data.append(tuple(temp))
        
        #initiate centroid
        self.ind2centroid={ind:center for ind,center in enumerate(random.sample(self.data,self.K))}
            
    def getDistance(self,point1,point2):
        return np.sqrt(sum([(point1[i]-point2[i])**2 for i in range(len(point1))]))
    
    # minDis function
    def getClosestCentroid(self,point):
        ind2distance={}
        for ind,center in self.ind2centroid.items():
            ind2distance[ind]=self.getDistance(point,center)
        return min(ind2distance,key=ind2distance.get)
    
    # getVar function
    def getCostValue(self,ind2cluster):
        total=0
        for ind,points in ind2cluster.items():
            total+=sum([self.getDistance(point,self.ind2centroid[ind]) for point in points])
        return total        
    
    def getNewCentroids(self,ind2cluster):
        ind2newcenter={}
        for ind,points in ind2cluster.items():
            xs,ys=zip(*points)
            ind2newcenter[ind]=(np.mean(xs),np.mean(ys))
        return ind2newcenter
    
    # main function
    def executeKMeans(self):
        costValues=[]
        while True:
            ind2cluster={ind:[] for ind in self.ind2centroid.keys()}
            for point in self.data:
                closestInd=self.getClosestCentroid(point)
                ind2cluster[closestInd].append(point)
                
            self.ind2centroid=self.getNewCentroids(ind2cluster)
            cost=self.getCostValue(ind2cluster)
            
            if len(costValues)>0 and abs(cost-costValues[-1])==0:
                self.ind2cluster=ind2cluster
                break
            else:
                costValues.append(cost)
                    
    def plotDataPoints(self,ax):
        cmap=cm.Paired
        for ind,points in self.ind2cluster.items():
            xs,ys=zip(*points)
            ax.scatter(xs,ys,color=cmap(ind/len(self.ind2cluster)),marker='.')
        ax.set_title("Seed=%d   Cost=%d"%(self.seed,self.getCostValue(self.ind2cluster)))
    
    def plotCentroid(self,ax):
        xs,ys=zip(*self.ind2centroid.values())
        ax.scatter(xs,ys,c='red',marker='X',s=100)


fig,axes=plt.subplots(2,3,figsize=(20,10))
for i,ax in enumerate(axes.flat):
    kmeans=Kmeans("k_means_data",6,i)
    kmeans.executeKMeans()
    kmeans.plotDataPoints(ax)
    kmeans.plotCentroid(ax)
    print(kmeans.ind2centroid)
plt.savefig("Figure.png")







