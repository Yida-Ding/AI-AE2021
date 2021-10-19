import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:
    def __init__(self,fileName):
        file=open("data/%s.txt"%fileName,"r") #laod data from txt file
        self.data=[]
        for string in file.readlines():
            temp=string[:-1].split(',')     #remove the last character "\n"
            self.data.append([1.]+[float(s) for s in temp])   #add the first column with 1s
        self.data=np.array(self.data)
        self.model=np.array([0.]*(len(self.data[0])-1))        #model is [theta0,theta1,theta2]
    
    #plot the sample data points
    def plotDataPoints(self,ax):
        admitt=self.data[np.where(self.data[:,-1]==1)[0]]  # admitted data points
        reject=self.data[np.where(self.data[:,-1]==0)[0]]  # rejected data points
        ax.scatter(admitt[:,1],admitt[:,2],c='green',marker='o',label='Admitted')
        ax.scatter(reject[:,1],reject[:,2],c='purple',marker='x',label='Rejected')
        ax.set_xlabel("Exam1")
        ax.set_ylabel("Exam2")
        ax.legend()
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    #the cost function
    def getFinalCost(self):
        hvalue=self.sigmoid(self.data[:,:-1]@self.model)
        return np.mean((-self.data[:,-1])*np.log(hvalue)-(1-self.data[:,-1])*np.log(1-hvalue))
    
    #train the model by gradient descent    
    def trainModel(self,alpha,epoch):
        self.alpha,self.epoch=alpha,epoch
        for t in range(epoch):
            for j in range(len(self.model)):
                errorSum=0            
                for i,row in enumerate(self.data):
                    errorSum+=(self.sigmoid(row[:-1]@self.model)-row[-1])*row[j]
                self.model[j]-=errorSum*alpha/len(self.data)
            print("Epoch %d"%t,"Current error:",errorSum)
    
    def plotRegressionLine(self,ax):
        xs=np.linspace(0,100,5)
        ys=[-(self.model[0]+x*self.model[1])/self.model[2] for x in xs]
        ax.plot(xs,ys,c='orange',label="Regression line: y=%.4fx+%.4f"%(-self.model[1]/self.model[2],-self.model[0]/self.model[2]))
        ax.set_title("Logistic Regression: alpha=%f, epoch=%d"%(self.alpha,self.epoch))
        print("Regression line: y=%.4fx+%.4f"%(-self.model[1]/self.model[2],-self.model[0]/self.model[2]))
        ax.legend()

    #predict the probability of admission for a student
    def predict(self,x1,x2):
        probability=self.sigmoid(np.array([1.,x1,x2])@self.model)
        return probability


if __name__=='__main__':
    LR=LogisticRegression("ex2data1")
    fig,ax=plt.subplots(1,1,figsize=(8,6),dpi=500)
    LR.plotDataPoints(ax)
    LR.trainModel(0.0019,100000)
    LR.plotRegressionLine(ax)
    print("The probability is",LR.predict(45,85))
    print("The final cost is",LR.getFinalCost())
    plt.savefig("Result1.png")




