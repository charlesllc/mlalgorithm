# -*- encoding=utf-8 -*-
'''
author: charlesllc
'''

import numpy as np
from copy import deepcopy

#Neural Network
'''
programming the neural network using forward propagation, back propagation and gradient descent.
'''
class neural_network(object):
    '''
    initializing hyperparameters:
    hidden_layers: int, the hidden layers
    hidden_units: tuple, the units of the hidden layers
    alpha: float, the learning rate
    lmbd: float, the regularization
    max_iter: int, the maximum iterating step
    _grad_check: bool, whether checking the gradient from backprop
    _min_dj: float, the gradient threshold below which the iteration is terminated.
    coef: dict, the model parameters; W: coefficients of independent variables; b: intercepts
    _loss: list, the losses of each iteration
    '''
    def __init__(self, hidden_layers=1, hidden_units=(3,), alpha=0.01, lmbd=0.1
                , max_iter=1000, grad_check=False, min_dj=0.0001):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.coef = {'W':None, 'b':None}
        self._loss = []
        self._grad_check = grad_check
        self._min_dj = min_dj

    # define the sigmoid function
    def sigmoid(self,x):
        """
        If z is very small(such as -50), the computer lets g(-z) equals zero because of the memory overflowing.
        so we replace z with -50. 
        """
        xc = deepcopy(x)
        xc[x<-50]=-50
        return 1 / (1 + np.exp(-xc))
    # define the derivation of sigmoid function
    def sigmoid_dev(self,x):
        sg = self.sigmoid(x)
        return sg * (1 - sg)

    # define the loss function of neural network
    def loss_func(self,W,b,X,y):
        L = len(W)
        m = X.shape[1]
        #forward propagation
        Z = [None for i in range(L)]
        A = [None for i in range(L+1)]
        A[0] = X
        for l in range(1, L+1):
            Z[l-1] = W[l-1].dot(A[l-1]) + b[l-1]
            A[l] = self.sigmoid(Z[l-1])
        y1prob = A[L][y==1]
        y0prob = 1-A[L][y==0]
        y0prob1_z = Z[L-1][y==0][y0prob==0]
        '''1 plus a very small value will be set 1. Thus yp=sigmoid(z) equals 1 when z is very large.
        In this situation, if the actual y=0 and the predicted yp=1, the loss term log(1-yp)=log(0) which is not defined.
        log(1-yp)=log(1-sigmoid(z))=log(1+exp(z))≈z when z very large. We can approximate log(1-yp) with z.
        '''
        neglogprobs = np.sum(-np.log(y1prob)) + np.sum(-np.log(y0prob[y0prob>0])) + np.sum(y0prob1_z)
        reg_loss = 0
        for l in range(1,L+1):
            reg_loss += self.lmbd/2*(np.sum(np.square(W[l-1])))
        loss = 1/m*(neglogprobs+reg_loss)
        return Z, A, loss

    def gradient_check(self,w,b,DW,X,y,epsilon=0.000000001):
        w[0][0][0] += epsilon
        Jup = self.loss_func(w,b, X, y)[2]
        w[0][0][0] -= 2*epsilon
        Jlow = self.loss_func(w,b, X, y)[2]
        w[0][0][0] += epsilon
        gradapprox = (Jup-Jlow)/(2*epsilon)
        return DW[0][0][0], gradapprox

    def fit(self, X, y):
        self._loss = []
        alpha = self.alpha
        lmbd = self.lmbd
        max_iter = self.max_iter
        n, m = X.shape
        L = self.hidden_layers + 1
        nl = (n,) + self.hidden_units + (1,)
        W = [None for i in range(L)]
        b = [None for i in range(L)]
        #initialize parameters W, b
        for l in range(1,L+1):
            W[l-1] = np.random.randn(nl[l], nl[l-1])  #/ np.sqrt(nl[l-1])
            b[l-1] = np.zeros((nl[l],1))
        DW = [None for x in range(L)]
        db = [None for x in range(L)]
        self.coef['W'] = W
        self.coef['b'] = b
        y = y.reshape(nl[L], m)
        counter = 0
        for i in range(0, max_iter):
            #forward propagation
            Z, A, loss = self.loss_func(W,b,X,y)
            self._loss.append(loss)
            if abs(self._loss[i-1]-self._loss[i])<self._min_dj:
                counter += 1
                if counter>100:
                    break
            #back propagation
            dlt = [None for x in range(L)]
            for l in range(L,0,-1):
                if l == L:
                    dlt[l-1] = A[l] - y
                else:
                    dlt[l-1] = ((W[l].T).dot(dlt[l])) * (self.sigmoid_dev(Z[l-1]))
            for l in range(1,L+1):
                DW[l-1] = 1/m * (dlt[l-1].dot(A[l-1].T) + lmbd*W[l-1])
                db[l-1] = 1/m * (np.sum(dlt[l-1], axis=1)).reshape(nl[l],1)
            #gradient check
            if self._grad_check and i<=20:
                print(self.gradient_check(W,b,DW,X,y))
            #update the parameters by gradient descent
            for l in range(1,L+1):
                W[l-1] += -alpha*DW[l-1]
                b[l-1] += -alpha*db[l-1]
        #save the trained model
        self.coef['W'] = W
        self.coef['b'] = b
        self.coef['DW'] = DW
        self.coef['db'] = db
    def predict(self,x):
        W = self.coef['W']
        b = self.coef['b']
        #forward propagation
        L = len(W)
        #forward propagation
        Z = [None for i in range(L)]
        A = [None for i in range(L+1)]
        A[0] = x
        for l in range(1, L+1):
            Z[l-1] = W[l-1].dot(A[l-1]) + b[l-1]
            A[l] = self.sigmoid(Z[l-1])
        return np.sign(np.sign(A[L]-0.5) + 1)

#model selection
#order selection
#(1,(3,)) is the best choice for the moons dataset
def model_selection_order(X,y,Xt,yt,hidden_layer={1:(1,(3,)),2:(1,(6,)),3:(2,(3,3)),4:(2,(4,4))}):

    max_iter = 20000
    loss_train = {}
    loss_test = {}
    model = {}
    for key, param in hidden_layer.items(): 
        L, units = param
        clf_nn = neural_network(hidden_layers=L, hidden_units=units,alpha=2, lmbd=0 \
                , max_iter=max_iter,grad_check=True, min_dj=0)
        clf_nn.fit(X.T,y.T)
        model[key] = clf_nn
        loss_train[key] = clf_nn._loss[max_iter-1]
        loss_test[key] = clf_nn.loss_func(clf_nn.coef['W'],clf_nn.coef['b'],Xt.T,yt.T)[2]
    return model, loss_train, loss_test

# reguliazation selection
def model_selection_lambda(X,y,Xt,yt,hidden_layer=(1,(3,)),lambd=[]):
    lmdb = {i:lambd[i-1] for i in range(1,len(lambd)+1)}
    max_iter = 20000
    loss_train = {}
    loss_test = {}
    model = {}
    L, units = hidden_layer
    for key, param in lmdb.items(): 
        clf_nn = neural_network(hidden_layers=L, hidden_units=units,alpha=2, lmbd=param \
                , max_iter=max_iter,grad_check=True, min_dj=0)
        clf_nn.fit(X.T,y.T)
        model[key] = clf_nn
        loss_train[key] = clf_nn._loss[max_iter-1]
        loss_test[key] = clf_nn.loss_func(clf_nn.coef['W'],clf_nn.coef['b'],Xt.T,yt.T)[2]
    return model, loss_train, loss_test


#Learning Curve
def model_selection_learning(X,y,Xt,yt,clf,lambd=0,M=[2,4,8,15,30,50,100,500]):
    m_d = {i:M[i-1] for i in range(1,len(M)+1)}
    loss_train = {}
    loss_test = {}
    model = {}
    for key, param in m_d.items():
        Xtr = X[range(param),:]
        ytr = y[range(param),:]
        clf.fit(Xtr,ytr)
        model[key] = clf
        loss_train[key] = clf.loss_func(clf.coef['W'],clf.coef['b'],Xtr,ytr)[2]
        loss_test[key] = clf.loss_func(clf.coef['W'],clf.coef['b'],Xt,yt)[2]
    return model, loss_train, loss_test
