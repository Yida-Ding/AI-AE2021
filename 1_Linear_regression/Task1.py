import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self,fileName,model):
        file=open("数据附件/%s.txt"%fileName,"r")
        self.data=[]
        for string in file.readlines():
            temp=string[:-1].split(',')     #remove the last character "\n"
            self.data.append((1,float(temp[0]),float(temp[1]))) 
        self.trainData=self.data[:int(len(self.data)*0.7)]      #70% data for training
        self.testData=self.data[int(len(self.data)*0.7):]
        self.model=model        #model is [theta0,theta1]

    def predict(self,xs):
        return self.model[0]*xs[0]+self.model[1]*xs[1]
        
    def calculateLoss(self,dataType="testData"):
        if dataType=="testData":
            return sum([(self.predict(row[:-1])-row[-1])**2 for row in self.trainData])/(2*len(self.trainData))
        elif dataType=="trainData":
            return sum([(self.predict(row[:-1])-row[-1])**2 for row in self.testData])/(2*len(self.testData))
        
    def trainModelByDelta(self,alpha,delta):
        self.alpha,self.delta=alpha,delta
        actdeltas=[float("inf")]*2
        iteration=0
        while max(actdeltas)>delta:
            actdeltas=[]
            for j in range(len(self.model)):
                actdelta=0            
                for i,row in enumerate(self.trainData):
                    actdelta+=(self.predict(row[:-1])-row[-1])*row[j]/len(self.trainData)
                actdeltas.append(abs(actdelta))
                self.model[j]-=actdelta*alpha
            
            iteration+=1
            print("Iteration:",iteration,"Current delta:",max(actdeltas))
            
    def trainModelByEpoch(self,alpha,epoch):
        self.alpha,self.epoch=alpha,epoch
        for t in range(epoch):
            for j in range(len(self.model)):
                errorSum=0            
                for i,row in enumerate(self.trainData):
                    errorSum+=(self.predict(row[:-1])-row[-1])*row[j]    
                self.model[j]-=errorSum*alpha/len(self.trainData)
                
            print("Epoch %d"%t,"Current error:",errorSum)
            
    def plotDataPoints(self,ax):
        datasets=[self.trainData,self.testData]
        colors=['green','purple']
        labels=['Training data','Testing data']
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        for i in range(len(datasets)):            
            L=list(zip(*datasets[i]))
            xs,ys=L[1],L[2]
            ax.scatter(xs,ys,c=colors[i],label=labels[i])
        ax.legend()
        
    def plotRegressionLine(self,ax,method):
        xs=np.linspace(4,30,5)
        ys=[self.model[0]+self.model[1]*x for x in xs]
        ax.plot(xs,ys,c='orange',label="Regression line: y=%.4fx+%.4f"%(self.model[0],self.model[1]))
        if method=="epoch":
            ax.set_title("Linear Regression: alpha=%f, epoch=%d"%(self.alpha,self.epoch))
        elif method=="delta":
            ax.set_title("Linear Regression: alpha=%f, delta=%f"%(self.alpha,self.delta))
        ax.legend()
        
        
if __name__=='__main__':
    LR=LinearRegression("ex1data1",[0,0])    
    LR.trainModelByDelta(alpha=0.01,delta=0.000001)
    print("Test data loss:",LR.calculateLoss("testData"))
    print("Test data loss:",LR.calculateLoss("trainData"))
    print("Model parameters: theta0=",LR.model[0],"theta1=",LR.model[1])
    print("Regression line: y=%.4fx+%.4f"%(LR.model[0],LR.model[1]))
    fig,ax=plt.subplots(1,1,figsize=(8,6),dpi=500)
    LR.plotDataPoints(ax)
    LR.plotRegressionLine(ax,"delta")
    plt.savefig("Result1.png")
