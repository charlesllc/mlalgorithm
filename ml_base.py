# -*- encoding=utf-8 -*-

from sklearn import linear_model, datasets
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.special import comb, perm
from itertools import combinations, permutations
from random import random
from math import ceil

#decision_boundary
def plot_decision_boundary(X,y,feat_engi_func,pred_func):
 
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    x1 = xx.ravel()
    x2 = yy.ravel()
    Xn = feat_engi_func(np.c_[x1, x2])
    # 用预测函数预测一下
    Z = np.array(pred_func(Xn))
    Z = Z.reshape(xx.shape)
 
    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], s=40, c=y.reshape(-1))
    plt.show()


#feature_engineering
class feature_engineering(object):
    def __init__(self,solver):
        self.solver = solver

    def polynomial_feature(self,X,p):
        m,n = X.shape
        all_comb = []
        for k in range(2,p+1):
            al = list(combinations(list(range(n))*k, k))
            al = [tuple(sorted(v)) for v in al]
            al = list(set(al))
            all_comb += al
        poly_ext = lambda v: np.prod(X[:,v], axis=1)
        Xp = np.array([poly_ext(v) for v in all_comb]).T

        return np.c_[X, Xp]

    def linear_kernel(self,X,Y):
        return X.dot(Y.T)

    def gaussian_distance(self,X,y,width):
        D = X-y
        return np.exp(-np.sum(D*D,axis=1)/(2*width*width))

    def gaussian_kernel(self,X,Y,width):
        mx,nx = X.shape
        my,ny = Y.shape
        Xn = np.zeros((mx,my))
        for i in range(my):
            Xn[:,i] = self.gaussian_distance(X,Y[i],width)
        return Xn

    def constant(self,X,*args):
        return X
    # getter
    @property
    def return_feature(self):
        if self.solver == 'gaussian_kernel':
            return self.gaussian_kernel
        elif self.solver == 'linear_kernel':
            return self.linear_kernel
        elif self.solver == 'polynomial':
            return self.polynomial_feature
        else:
            print('Nothing Changed')
            return self.constant
#fe = feature_engineering('woe')
#Xn = fe.return_feature(X,y,navalid=True)
#bin_res = fe.bin_res

#loss functions
class loss_func_class(object):
    def __init__(self,model_type):
        self.model_type = model_type

    def regularization(self,W,penalty):
        if penalty == 'l1':
            Jw, DJw =np.sum(np.abs(W)), np.sign(W)
        elif penalty == 'l2':
            Jw, DJw = 1/2 * W.dot(W.T), W
        else:
            Jw, DJw = 0, np.zeros(W.shape)
        return Jw, DJw

    def sgn(self,x):
        return np.sign(np.sign(x) - 1) + 1

    def loss_svm_hinge(self,W,b,X,y,C,penalty):
        m,n = X.shape
        y = y.reshape(m,1)
        z = X.dot(W.T) + b
        Jw, DJw = self.regularization(W,penalty)
        loss = C * ((y.T).dot((1-z)*self.sgn(1-z)) + ((1-y).T).dot((1+z)*self.sgn(1+z))) + Jw
        '''
        Here, a loss divided by m is smoother than the one which isn't divided by m, which means a more suitable DW/db
        for the gradient descent algrithm and will not change the optimization point.
        '''
        loss = loss/m
        delta = C * (-y*self.sgn(1-z) + (1-y)*self.sgn(1+z)).T
        DW = (delta.dot(X) + DJw) / m
        db = np.sum(delta) / m
        return DW,db,loss[0,0]

    def loss_constant(self):
        return 1

    def return_loss_func(self):
        if self.model_type == 'neural_networl':
            func = self.loss_neural_network
        elif self.model_type == 'logistic_regression':
            func = self.loss_logistic_regression
        elif self.model_type == 'svm_hinge':
            func = self.loss_svm_hinge
        else:
            func = loss_constant
        return func

#loss_func = loss_func_class('svm_hinge').return_loss_func()

# Algorithm evaluation
def gradient_check(loss_func,w,b,DW,X,y,epsilon=0.000000001):
    switcher = isinstance(w,list)
    if switcher:
        w[0][0][0] += epsilon
    else:
        w[0][0] += epsilon
    Jup = loss_func(w,b, X, y)[2]
    if switcher:
        w[0][0][0] -= 2*epsilon
    else:
        w[0][0] -= 2*epsilon
    Jlow = loss_func(w,b, X, y)[2]
    if switcher:
        w[0][0][0] += epsilon
    else:
        w[0][0] += epsilon
    gradapprox = (Jup-Jlow)/(2*epsilon)
    dw = DW[0][0][0] if switcher else DW[0][0]
    return dw, gradapprox

def plot_loss_of_one_coef(model,loss_func,X,y,w=(1,1,1),times=10):
    l,i,j = w
    w = deepcopy(model['W'])
    b = deepcopy(model['b'])
    if isinstance(w,list):
        tmp = w[l-1][j-1][i-1].copy()
    else:
        tmp = w[j-1][i-1].copy()
    loss_m = loss_func(w,b,X,y)[2]
    npa = np.linspace(tmp-times*abs(tmp), tmp+times*abs(tmp), 1000)
    loss = []
    for k in range(1000):
        if isinstance(w,list):
            w[l-1][j-1][i-1] = npa[k]
        else:
            w[j-1][i-1] = npa[k]
        try:
            loss.append(loss_func(w,b,X,y)[2])
        except:
            print()
    print(tmp,loss_m)
    plt.scatter(tmp,loss_m,marker='o')
    plt.plot(npa,loss)
    plt.show()
    

def plot_loss_of_two_coef(model,loss_func,X,y,w1=(1,1,1),w2=(1,2,1),grid_size=100,times=10):
    w = deepcopy(model['W'])
    b = deepcopy(model['b'])
    if isinstance(w,list):
        tmpx = w[w1[0]-1][w1[2]-1][w1[1]-1].copy()
        tmpy = w[w2[0]-1][w2[2]-1][w2[1]-1].copy()
    else:
        tmpx = w[w1[2]-1][w1[1]-1].copy()
        tmpy = w[w2[2]-1][w2[1]-1].copy()
    loss_m = loss_func(w,b,X,y)[2]
    mx = np.linspace(tmpx-times*abs(tmpx), tmpx+times*abs(tmpx), grid_size)
    my = np.linspace(tmpy-times*abs(tmpy), tmpy+times*abs(tmpy), grid_size)
    xx, yy = np.meshgrid(mx,my)
    loss = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if isinstance(w,list):
                w[w1[0]-1][w1[2]-1][w1[1]-1] = mx[i]
                w[w2[0]-1][w2[2]-1][w2[1]-1] = my[j]
            else:
                w[w1[2]-1][w1[1]-1] = mx[i]
                w[w2[2]-1][w2[1]-1] = my[j]
            loss[j,i] = loss_func(w,b,X,y)[2]
    
    print(tmpx,tmpy,loss_m)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,loss,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.scatter(tmpx,tmpy,loss_m,c='red',marker='o')
    plt.show()

#plot_loss_of_two_coef(clf_nn.coef, clf_nn.loss_func,X,y,w1=(1,1,1),w2=(1,2,1),grid_size=100,times=30)
#plot_loss_of_one_coef(clf_nn.coef, clf_nn.loss_func,X,y,w=(1,2,1),times=20)

#---------Model_selection--
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

#--------Model_evaluation---------
class model_evaluation(object):
    def __init__(self,y,yprob):
        self.y = y.T
        self.yprob = yprob.T
        self.error_rate = {}
        self.tpr = {}  #recall
        self.fpr = {}
        self.precision = {}
        self.auc = 0
        self.gini = 0
        self.f1 = {}
        self._lorenz = []
        self.thds = np.round(np.linspace(0,1,1001), 3)
        for thd in self.thds:
            ypre = np.sign(self.yprob +0.0000001 - thd)
            ypre[ypre==-1] = 0
            self.error_rate[thd] = np.sum(np.not_equal(ypre, self.y)) / self.y.shape[1]
            tp = np.sum(ypre[self.y==1])
            fp = np.sum(ypre[self.y==0])
            fn = np.sum(1-ypre[self.y==1])
            tn = np.sum(1-ypre[self.y==0])
            self.tpr[thd] = tp/(tp+fn)
            self.fpr[thd] = fp/(fp+tn)
            self.precision[thd] = tp/(tp+fp) if tp+fp>0 else 1
            self.f1[thd] = 2 * self.precision[thd] * self.tpr[thd] / (self.precision[thd] + self.tpr[thd])

        for i in range(len(self.thds)-1,0,-1):
            delta_auc = self.fpr[self.thds[i-1]] - self.fpr[self.thds[i]]
            self.auc += delta_auc*self.tpr[self.thds[i-1]]
            
        ptils = [i for i in range(101)]
        yprob_pt = np.percentile(self.yprob, ptils)
        for i in range(101):
            thd = yprob_pt[i]
            self._lorenz.append(np.sum(self.y[self.yprob<=thd]) / np.sum(self.y))
            if i > 0:
                self.gini += 0.01 * self._lorenz[i-1]
        self.gini = (1 - 2*self.gini)

        
        self.ks = np.max(np.array(list(self.tpr.values()))-np.array(list(self.fpr.values())))

        self.lift = {}
        for thd in np.round(np.linspace(0.9,0,10), 3):
            ypart = self.y[self.yprob>thd]
            self.lift[thd] = (np.sum(ypart) / ypart.shape[0]) / (np.sum(self.y) / self.y.shape[1])

        self.single_swither = True
    def lorenz_curve(self):
        x = ptils = [i/100 for i in range(101)]
        y = self._lorenz
        plt.plot(x,x,'-.',c='black')
        plt.plot(x,y)
        plt.xlabel('Cumulative share of sample from lowest to highest predicted probability')
        plt.ylabel('Cumulative share of true positive')
        plt.title('Lorenz Curve: gini=%.2f' % self.gini)
        plt.legend()
        if self.single_swither:
            plt.show()  

    def lift_curve(self):
        x = [str(i) for i in self.lift.keys()]
        y = list(self.lift.values())
        benchmark = (np.sum(self.y) / self.y.shape[1]) * np.ones(len(x))
        lift_max = 1/benchmark
        plt.plot(x,benchmark,'-.',c='black', label='the positive rate')
        plt.plot(x,lift_max,'-.',c='red', label='the upper bound')
        plt.plot(x,y,label='Lift')
        plt.xlabel('Cutoff')
        plt.ylabel('Lift')
        plt.title('Lift Curve')
        plt.legend()
        if self.single_swither:
            plt.show()        

    def cut_errorrate_curve(self):
        '''
        draw misclassification rate verse cutoff 
        '''
        x = list(self.thds)
        y = list(self.error_rate.values())
        plt.plot(x,y)
        for a,b in self.error_rate.items():
            if (a*1000) % 100 == 0:
                plt.text(a,b+0.01, '%.3f' % b)
        plt.xlabel('Cutoff')
        plt.ylabel('ErrorRate')
        plt.title('ErrorRate-Cutoff Curve')
        if self.single_swither:
            plt.show()

    def roc_curve(self):
        '''
        draw ROC(tpr-fpr) curve. We want to increase tpr while keeping fpr(the type2error) small by decreasing the cut-off probability.
        '''
        x = list(self.fpr.values())
        y = list(self.tpr.values())
        plt.plot(x,y)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC(tpr-fpr) curve: AUC=%.2f' % self.auc)
        if self.single_swither:
            plt.show()

    def ks_curve(self):
        '''
        draw ks_curve: tpr-cutoff and fpr-cutoff curve with cutoff decreasing from 1 to 0.
        '''
        x = self.thds
        y1 = list(self.tpr.values())
        y2 = list(self.fpr.values())
        y3 = list(self.tpr.keys())[np.argmax(np.array(y1)-np.array(y2))]

        plt.plot(x,y1,c='green',label='tpr: the cumulative share of the positive')
        plt.plot(x,y2,c='red',label='fpr: the cumulative share of the negtive')
        plt.axvline(y3,ls='-.',c='grey')
        plt.title('KS curve: ks=%.2f' % self.ks)
        plt.xlabel('Cutoff')
        plt.ylabel('%')
        plt.legend()
        if self.single_swither:
            plt.show()

    def pr_curve(self):
        '''
        draw precision-recall curve
        '''
        recall = self.tpr
        x = list(recall.values())
        y = list(self.precision.values())
        plt.plot(x,y)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve')
        if self.single_swither:
            plt.show()


    def f1_curve(self):
        '''
        draw F1-Score curve
        '''
        x = self.thds
        y_f1 = list(self.f1.values())
        y_precision = list(self.precision.values())
        y_recall = list(self.tpr.values())
        plt.plot(x,y_f1,c='red',label='F1 Score')
        plt.plot(x,y_precision,c='blue',label='Precision')
        plt.plot(x,y_recall,c='green',label='Recall')
        plt.title('F1 Score Curve')
        plt.xlabel('Cutoff')
        plt.ylabel('F1 Score')
        plt.legend()
        if self.single_swither:
            plt.show()

    def yprob_distribution(self):
        plt.hist(self.yprob[self.y==1],bins=50,normed=True,label='pos')
        plt.hist(self.yprob[self.y==0],bins=50,normed=True,label='neg')
        plt.title('the distribution of yprob over the label')
        plt.xlabel('yprob')
        plt.ylabel('the frequency(%)')
        plt.legend()
        if self.single_swither:
            plt.show()


    def all_curve(self):
        self.single_swither = False
        plt.subplot(2,3,1)
        self.cut_errorrate_curve()
        plt.subplot(2,3,2)
        self.roc_curve()
        plt.subplot(2,3,3)
        self.ks_curve()
        plt.subplot(2,3,4)
        self.pr_curve()
        plt.subplot(2,3,5)
        self.lift_curve()
        plt.subplot(2,3,6)
        self.yprob_distribution()
        plt.show()
        self.single_swither = True
