# -*- encoding=utf-8 -*-

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.pyplot as plt
from random import random

#perceptron
class perceptron(object):
    def __init__(self,eta=0.5,max_iter=100,draw_boundary=False):
        self.eta = eta
        self.coef = {}
        self._loss = []
        self.max_iter = max_iter
        self.draw_boundary = draw_boundary

    def loss_func(self,W,b,X,y):
        m = X.shape[0]
        innerp = X.dot(W.T) + b
        loss = np.sum(innerp/(W.dot(W.T)) * (np.sign(np.sign(innerp + 1)) - y))
        return loss/m

    def plot_cur_db(self,W,b,X,y,c='black'):
        # 设定最大最小值，附加一点点边缘填充
        x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
        x_grid = np.linspace(x_min,x_max,100)
        y_grid = -(1/W[0,1]) * (W[0,0]*x_grid + b)
        # 然后画出图
        plt.plot(x_grid,y_grid,c=c)
        plt.scatter(X[:,0], X[:,1], s=40, c=y)
        plt.pause(1)


    def fit(self,X,y):
        m,n = X.shape

        W = np.zeros((1,n))
        b = random()

        flag = True
        ite = 0
        while flag and ite<self.max_iter:
            count = 0
            if self.draw_boundary:
                self.plot_cur_db(W,b,X,y,c='grey')
            for i in range(m):
                x = X[i]
                yp = 1 if x.dot(W.T) + b >= 0 else 0  

                if yp != y[i]:
                    count += 1
                    W += self.eta * x * (y[i]-yp)
                    b += self.eta * (y[i]-yp)
            if count == 0:
                flag = False

            ite += 1

        self.coef['W'] = W
        self.coef['b'] = b
        self.plot_cur_db(self.coef['W'],self.coef['b'],X,y,c='red')
        print('Finished')

#train a perceptron for the blobs
#m = 700
#X, y = sklearn.datasets.make_blobs(m,centers=2,cluster_std=0.5,center_box=(0,5))
#plt.scatter(X[:,0], X[:,1], s=40, c=y)
#plt.show()
#clf_perp = perceptron(eta=0.1,max_iter=10000000,draw_boundary=True)
#clf_perp.fit(X,y)

#######################################
#SVM
#SVM algorithm
class svm(object):
    def __init__(self,C=0.5,loss_type='svm_hinge',max_iter=100,alpha=0.1,penalty='l2',draw_boundary=False):
        self.C = C
        self.coef = {}
        self.max_iter = max_iter
        self.draw_boundary = draw_boundary
        self.alpha = alpha
        self.penalty = penalty
        #the loss function of SVM is defined in ml_base
        loss_func_t = loss_func_class(loss_type).return_loss_func()
        self.loss_func = lambda W,b,X,y: loss_func_t(W,b,X,y,self.C,self.penalty)

    def plot_cur_db(self,W,b,X,y,c='black'):
        m,n = X.shape
        x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
        x_grid = np.linspace(x_min,x_max,100)
        y_grid = -(1/W[0,1]) * (W[0,0]*x_grid + b)
        # 然后画出图
        plt.plot(x_grid,y_grid,c=c)
        plt.scatter(X[:,0], X[:,1], s=40, c=y.reshape(m))
        plt.pause(0.1)


    def fit(self,X,y):
        m,n = X.shape
        #W = deepcopy(W)
        #b = deepcopy(b)
        W = np.random.randn(1,n)
        b = random()
        self._loss = []
        for i in range(self.max_iter):
            if self.draw_boundary and i % 1000 == 0:
                self.plot_cur_db(W,b,X,y,c='grey')
            DW,db,loss = self.loss_func(W,b,X,y)

            W += -self.alpha * DW
            b += -self.alpha * db
            self._loss.append(loss)

        self.coef['W'] = W
        self.coef['b'] = b
        self.plot_cur_db(self.coef['W'],self.coef['b'],X,y,c='red')
        print('Finished')

    def sigmoid(self,x):
        """
        If z is very small(such as -50), the computer lets g(-z) equals zero because of the memory overflowing.
        so we replace z with -50. 
        """
        xc = deepcopy(x)
        xc[x<-50]=-50
        return 1 / (1 + np.exp(-xc))

    def scaling(self,x):
        a = np.max(x)
        b = abs(np.min(x))
        srange = a if a>=b else b
        return 10 * x / srange

    def predict_proba(self,X):
        W = self.coef['W']
        b = self.coef['b']
        R = (X.dot(W.T) + b) / np.sqrt(W.dot(W.T))
        return self.sigmoid(X.dot(W.T) + b)

    def predict(self,X):
        proba = self.predict_proba(X)
        return np.sign(np.sign(proba-0.5) + 1)


#train a SVM for the moons
#moons
#m = 700
#X, y = sklearn.datasets.make_moons(m,noise=0.1)
#y = y.reshape(m,1)
#plt.scatter(X[:,0], X[:,1], s=40, c=y[:,0])
#plt.show()
##test set
#mt=300
#Xt, yt = sklearn.datasets.make_moons(mt,noise=0.4)
#yt = yt.reshape(mt,1)
##feature_engineering
#n = 17
#Xf = X[0:n,:]
#Xn = feature_engineering('gaussian_kernel').return_feature(X,Xf,0.2)
#Xtn = feature_engineering('gaussian_kernel').return_feature(Xt,Xf,0.2)
##train
#clf_svm_hinge = svm(C=100,loss_type='svm_hinge',max_iter=60000,alpha=0.0005,penalty='l2',draw_boundary=False)
#plt.subplot(1,2,1)
#clf_svm_hinge.fit(Xn,y.reshape(m,1))
#plt.subplot(1,2,2)
#plt.plot(clf_svm_hinge._loss)
#plt.show()