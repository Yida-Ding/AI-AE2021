import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAagent:
    def __init__(self,fileName):
        file=open("%s.txt"%fileName,"r")
        self.data=[]
        for string in file.readlines():
            temp=string[:-1].split('.')
            self.data.append([int(s) for s in temp])
    
    def PCA(self):
        x=StandardScaler().fit_transform(self.data)
        pca_data = PCA(n_components=2)
        principalComponents_data = pca_data.fit_transform(x)
        principal_data_Df = pd.DataFrame(data = principalComponents_data
             , columns = ['principal component 1', 'principal component 2'])
        
        print(principal_data_Df)
        print('Explained variation per principal component: {}'.format(pca_data.explained_variance_ratio_))        
        return principal_data_Df
        
    def plotPCAData(self,principal_data_Df):
        plt.figure(figsize=(10,10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel('Principal Component - 1',fontsize=20)
        plt.ylabel('Principal Component - 2',fontsize=20)
        plt.title("Principal Component Analysis",fontsize=20)
        plt.scatter(principal_data_Df.loc[:,'principal component 1']
                   , principal_data_Df.loc[:,'principal component 2'], c = 'g', s = 50)
        
        
agent=PCAagent("pca_data")
principal_data_Df=agent.PCA()
agent.plotPCAData(principal_data_Df)
    







#def plotDataPoints():
#    fig = plt.figure(figsize=(8,6))
#    ax = fig.add_subplot(111, projection='3d')
#    colors=['green','purple']
#    ax.set_xlabel("X")
#    ax.set_ylabel("Y")
#    ax.set_zlabel("Z")
#    for i in range(len(datasets)):            
#        L=list(zip(*datasets[i]))
#        xs,ys,zs=L[1],L[2],L[3]
#        ax.scatter(xs,ys,zs,c=colors[i],label=labels[i],marker='o',s=50)
#    ax.legend()
#        
#if __name__=='__main__':
#    LRM=LinearRegressionMatrix("ex1data2")
#    LRM.getModel()
#    print("Test data loss:",LRM.calculateLoss("testData"))
#    print("Test data loss:",LRM.calculateLoss("trainData"))
#    
#    fig = plt.figure(figsize=(8,6))
#    ax = fig.add_subplot(111, projection='3d')
#    LRM.plotDataPoints(ax)
#    LRM.plotRegressionSurface(ax)
#    plt.tight_layout()
#    plt.savefig("Result2.png")
#    
    
    
    
    
    

