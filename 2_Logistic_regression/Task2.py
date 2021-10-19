import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:
    def __init__(self,fileName):
        file=open("data/%s.txt"%fileName,"r") #load data from txt file
        self.data=[]
        for string in file.readlines():
            if string!='\n':    #check whether the line is empty or not
                temp=string[:-1].split(',')     #remove the last character "\n"
                self.data.append([1.]+[float(s) for s in temp])   #add the first column with 1s
        self.data=np.array(self.data)
        self.data=np.insert(self.data,3,(self.data[:,1])**2,axis=1) #insert a column of x^2
        self.data=np.insert(self.data,4,(self.data[:,2])**2,axis=1) #insert a column of y^2
        self.model=np.array([0.]*(len(self.data[0])-1))        #model is [theta0,theta1,theta2,theta3,theta4]
    
    #plot the sample data points
    def plotDataPoints(self,ax):
        admitt=self.data[np.where(self.data[:,-1]==1)[0]]  # data points that pass the test
        reject=self.data[np.where(self.data[:,-1]==0)[0]]  # data points that fail the test
        ax.scatter(admitt[:,1],admitt[:,2],c='green',marker='o',label='Pass')
        ax.scatter(reject[:,1],reject[:,2],c='purple',marker='x',label='Fail')
        ax.set_xlabel("microchip_test_1")
        ax.set_ylabel("microchip_test_2")
        ax.legend()
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    #train the model by gradient descent with normalization
    def trainModel(self,alpha,epoch,lambd):
        self.alpha,self.epoch,self.lambd=alpha,epoch,lambd
        for t in range(epoch):
            for j in range(len(self.model)):
                if j==0:        #for theta0
                    errorSum=0            
                    for i,row in enumerate(self.data):
                        errorSum+=(self.sigmoid(row[:-1]@self.model)-row[-1])*row[j]
                    self.model[j]-=errorSum*alpha/len(self.data)
                else:           #for other thetas
                    errorSum=lambd*self.model[j]  #the lambda*theta_j term
                    for i,row in enumerate(self.data):
                        errorSum+=(self.sigmoid(row[:-1]@self.model)-row[-1])*row[j]
                    self.model[j]-=errorSum*alpha/len(self.data)
                    
            print("Epoch %d"%t,"Current error:",errorSum)
    
    #plot the regression boundary with matplotlib contour function
    def plotRegression(self,ax):
        xrange=np.arange(-2,2,0.025)
        yrange=np.arange(-2,2,0.025)
        x,y=np.meshgrid(xrange, yrange)
        equation=self.model[0]+self.model[1]*x+self.model[2]*y+self.model[3]*(x**2)+self.model[4]*(y**2)
        ax.contour(x,y,equation,[0],colors='orange')        
        functionName="$%.4f+%.4fx+%.4fy%.4fx^2%.4fy^2=0$"%(self.model[0],self.model[1],self.model[2],self.model[3],self.model[4])
        ax.set_title("Logistic Regression:alpha=%f,epoch=%d,lambda=%d\n"%(self.alpha,self.epoch,self.lambd)+functionName)
        ax.legend()
        print(functionName)

if __name__=='__main__':
    LR=LogisticRegression("ex2data2")
    fig,ax=plt.subplots(1,1,figsize=(8,6),dpi=500)
    LR.plotDataPoints(ax)
    LR.trainModel(0.05,10000,1)
    LR.plotRegression(ax)
    plt.savefig("Result2.png")



