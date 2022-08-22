import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def opendata(textname):
    data=sio.loadmat(textname)
    X=data['X']
    return data,X

def estimate_parameters_for_gaussian_distribution(X):
    mu=np.mean(X,axis=0)
    sigma2=np.var(X,axis=0)
    return mu,sigma2

def gaussian_distribution(X,mu,sigma2):
    p=(1/np.sqrt(2*np.pi*sigma2))*np.exp(-(X-mu)**2/(2*sigma2)) # the (prob1,prob2) for many points
    return np.prod(p,axis=1) # prob for many points

def visualize_contours(mu,sigma2):
    x=np.linspace(5,25,100)
    y=np.linspace(5,25,100)
    xx,yy=np.meshgrid(x,y)
    X=np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)),axis=1)
    z=gaussian_distribution(X,mu,sigma2).reshape(xx.shape)
    cont_levels=[10**h for h in range(-20,0,3)]
    plt.contour(xx,yy,z,cont_levels)

def error_analysis(yp,yt):
    tp,fp,fn,tn=0,0,0,0
    for i in range(len(yp)):
        if yp[i]==yt[i]:
            if yp[i]==1:
                tp+=1
            else:
                tn+=1
        else:
            if yp[i]==1:
                fp+=1
            else:
                fn+=1
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return f1

def select_threshold(yval,pval):
    epsilons=np.linspace(min(pval),max(pval),1000)
    l=np.zeros((1,2))
    for e in epsilons: # for each epislon, there will be an evaluation f1
        ypre=(pval<e).astype(float)
        f1=error_analysis(ypre, yval)
        l=np.concatenate((l,np.array([[e,f1]])),axis=0)
    index = np.argmax(l[..., 1])
    return l[index, 0], l[index, 1]

def detection(X,e,mu,sigma2):
    p=gaussian_distribution(X,mu,sigma2)
    anomaly_points = np.array([X[i] for i in range(len(p)) if p[i]<e])
    return anomaly_points

def visualize_dataset(X):
    plt.scatter(X[..., 0], X[..., 1], marker='x', label='point')
    
def circle_anomaly_points(X):
    plt.scatter(X[..., 0], X[..., 1], s=80, facecolors='none', edgecolors='r', label='anomaly point')


def main(textname):
    data,X=opendata(textname) #X is the training set without y label, the goal is to train the model (miu, sigma2)
    X=np.vstack([X,[10,10],[9,9],[8,8]])
    visualize_dataset(X)
    mu,sigma2=estimate_parameters_for_gaussian_distribution(X)
    visualize_contours(mu, sigma2)
    Xval = data['Xval'] #cross validation dataset, with both x and y label, this will calcutate the error and thus epislon and F1
    yval = data['yval']
    e, f1 = select_threshold(yval.ravel(), gaussian_distribution(Xval, mu, sigma2)) #the goal is to find the best epsilon with largest F1
    anomaly_points = detection(X, e, mu, sigma2) # like testing
    circle_anomaly_points(anomaly_points)
    plt.title('anomaly detection')
    plt.legend()
    plt.show()
    print('e= ',e,", f1= ",f1)
    
main("ex8data1.mat")


