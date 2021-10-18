import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressionMatrix:
    def __init__(self,fileName):
        file=open("数据附件/%s.txt"%fileName,"r") #load data from txt file
        self.data=[]
        for string in file.readlines():
            temp=string[:-1].split(',')    # remove the last character "\n"
            self.data.append([1]+[float(d) for d in temp])
        self.trainData=np.array(self.data[:int(len(self.data)*0.95)])  # 95% data for training
        self.testData=np.array(self.data[int(len(self.data)*0.05):]) 
    
    def getModel(self):
        X=self.trainData[:,:-1]
        y=self.trainData[:,-1]
        try:
            temp=np.linalg.inv(np.dot(X.transpose(),X))
            self.model=np.dot(np.dot(temp,X.transpose()),y) # the equation to obtain the model parameters
            print("Linear regression by matrix operation\n Model: z=%d+%dx+%dy"%(self.model[0],self.model[1],self.model[2]))
        
        except: # matrix X may be singular or in other extreme case 
            print("Matrix X is invalid !")
        
    def predict(self,xs):
        return self.model[0]*xs[0]+self.model[1]*xs[1] # get prediction value from model 
    
    def calculateLoss(self,dataType="testData"):
        if dataType=="testData":
            return sum([(self.predict(row[:-1])-row[-1])**2 for row in self.trainData])/(2*len(self.trainData)) # the loss of test data
        elif dataType=="trainData":
            return sum([(self.predict(row[:-1])-row[-1])**2 for row in self.testData])/(2*len(self.testData)) # the loss of train data

    def plotDataPoints(self,ax): # plot the data points in the dataset, distinguished by training data and testing data
        datasets=[self.trainData,self.testData]
        colors=['green','purple']
        labels=['Training data','Testing data']
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        for i in range(len(datasets)):            
            L=list(zip(*datasets[i]))
            xs,ys,zs=L[1],L[2],L[3]
            ax.scatter(xs,ys,zs,c=colors[i],label=labels[i],marker='o',s=50)
        ax.legend()
        
    def plotRegressionSurface(self,ax):  # plot the regression line given the model parameters
        xs=np.linspace(0,5000,10)
        ys=np.linspace(0,6,10)
        X,Y=np.meshgrid(xs,ys)
        ax.plot_surface(X,Y,Z=self.model[0]+self.model[1]*X+self.model[2]*Y,color='orange',alpha=0.7)       
        ax.set_title("Linear regression by matrix operation\n Model: z=%d+%dx+%dy"%(self.model[0],self.model[1],self.model[2]),fontsize=10)
        ax.legend()

if __name__=='__main__':
    LRM=LinearRegressionMatrix("ex1data2")
    LRM.getModel()
    print("Test data loss:",LRM.calculateLoss("testData"))
    print("Test data loss:",LRM.calculateLoss("trainData"))
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    LRM.plotDataPoints(ax)
    LRM.plotRegressionSurface(ax)
    plt.tight_layout()
    plt.savefig("Result2.png")











