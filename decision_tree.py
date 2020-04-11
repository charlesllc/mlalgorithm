# -*- coding=utf-8 -*- 

import numpy as np
from math import log
from pandas.api.types import is_string_dtype

class decision_tree(object):
    def __init__(self,max_depth=5,split_rule='ID3'):
        self.max_depth = max_depth
        self.split_rule = split_rule
        self.model = {}

    # compute entropy
    def entropy(self,y):
        if isinstance(y, pd.Series):
            m = len(y)
            p = y.groupby(y).count() / m
            if self.split_rule=='ID3':
                pure = np.sum(-p * np.log(p)) / log(2)
            elif self.split_rule=='GINI':
                pure = np.sum(p * (1-p))
            else:
                pure = np.sum(-p * np.log(p)) / log(2)
            return pure
        else:
            p = y.div(y.sum(axis=1), axis=0)
            if self.split_rule=='ID3':
                pure = np.sum(-p*np.log(p),axis=1) / log(2)
            elif self.split_rule=='GINI':
                pure = np.sum(p*(1-p),axis=1)
            else:
                pure = np.sum(-p*np.log(p),axis=1) / log(2)
            return pure

    #compute the conditional entropy
    def con_entropy(self,x,y):
        if is_string_dtype(x):
            tmp = y.groupby(x).agg(['count',self.entropy])
            prior_prob = tmp['count'] / tmp['count'].sum()
            post_ent = tmp['entropy']
            con_ent = np.sum(prior_prob*post_ent)
            cutoff = list(prior_prob.index)
            prior_prob.index = list(range(len(prior_prob)))
        
        #optimize the split point of a continuous variable
        else:
            x.name = 1
            ptil = np.percentile(x,range(5,100,5))
            cutoffs = sorted(set([-np.inf] + list(ptil) + [np.inf]))
            labels = np.array([i for i in range(len(cutoffs)-1)])
            xs = pd.cut(x,cutoffs,labels=labels)
            post_cnt = y.groupby([y,xs]).count().unstack(level=0)
            post_cnt['prior'] = post_cnt.sum(axis=1)
            post_cnt_cum = post_cnt.cumsum()
            post_cnt_cum_op = post_cnt_cum.iloc[-1]-post_cnt_cum
            prior_P0 = post_cnt_cum.iloc[0:-1,-1]/post_cnt_cum.iloc[-1,-1]
            prior_P1 = post_cnt_cum_op.iloc[0:-1,-1]/post_cnt_cum.iloc[-1,-1]
            post_ent0 = self.entropy(post_cnt_cum.iloc[0:-1,0:-1])
            post_ent1 = self.entropy(post_cnt_cum_op.iloc[0:-1,0:-1])
            con_ents = prior_P0.mul(post_ent0,axis=0) + prior_P1.mul(post_ent1,axis=0)
            #choose the optimal threshold
            k = list(con_ents.index[con_ents==con_ents.min()])[0]
            con_ent,cutoff = con_ents.loc[k], cutoffs[k+1]
            prior_prob = pd.Series([prior_P0.loc[k],prior_P1.loc[k]])
        return con_ent,cutoff,prior_prob

    #generate the split node
    def split_node(self,X,y):
        res = X.apply(lambda x: con_entropy(x,y))
        pures = pd.Series([v[0] for v in res], index=res.index)
        if self.split_rule=='C4.5':
            gain = self.entropy(y) - pures
            prior_prob = pd.DataFrame([v[2] for v in res], index=res.index)
            ivalue = entropy(prior_prob)
            gain_ratio = gain.div(ivalue, axis=0)
            k = gain_ratio.index[gain_ratio==gain_ratio.max()][0]
        else:
            k = pures.index[pures==pures.min()][0]
        return k, res.loc[k][1]


    '''
    save tree as a json string.
    model = {'ypre':,'node':,'con':,'snode':[{'ypre':,'node':,'con':,'snode':}
                ,{'ypre':,'node':,'con':,'snode':[{'ypre':,'node':,'con':,'snode':},
                    {'ypre':},]
                    }
                ,
                ]
            }
    '''
    # train decision tree by recursion
    def __fit_base(self,X,y,model={},depth=0):
        ycnt = y.groupby(y).count()
        ypre = list(ycnt[ycnt==ycnt.max()].index)
        model['ypre'] = ypre[0]
        if depth < self.max_depth-1 and len(ycnt) > 1:
            split,cutoff = self.split_node(X,y)
            model['node'] = split
            is_string = isinstance(cutoff, list)
            xs= X[split] if is_string else np.sign(np.sign(X[split]-cutoff)-1)
            Xs = [item for _,item in X.groupby(xs)]
            ys = [item for _,item in y.groupby(xs)]
            model['con'] = {}
            model['con']['is_str'] = is_string
            model['con']['bins'] = cutoff if is_string else [-np.inf, cutoff, np.inf]
            #model['confunc'] = [lambda x:x==v for v in cutoff] if is_list else [lambda x:x<=cutoff, lambda x:x>cutoff]
            model['snode'] = [{} for v in range(len(ys))]
            for l in range(len(ys)):
                self.__fit_base(Xs[l],ys[l], model=model['snode'][l], depth=depth+1)

    def fit(self,X,y):
        m,n=X.shape
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)
        else:
            self.columns_o = {k:v for v,k in zip(X.columns,range(n))}
            X.columns = list(range(n))
        self.__fit_base(X,y,model=self.model,depth=0)

    '''predict'''
    def __predict_base(self,X,y=np.array([]),model={}):
        m,n = X.shape
        ypre = pd.Series(model['ypre'] * np.ones(m), index=X.index)
        if len(y)>0:
            model['err'] = np.sum(ypre != y)
        if 'node' not in model:
            #ypre = pd.Series(model['ypre'] * np.ones(m), index=X.index)
            self._ypre.append(ypre)
        else:            
            node = model['node']
            is_str, bins = model['con']['is_str'], model['con']['bins']
            label = bins if is_str else np.array(range(len(bins)-1))
            xs = X[node] if is_str else pd.cut(X[node],bins,labels=label)
            Xs = {k:v for k,v in X.groupby(xs)}
            ys = {k:v for k,v in y.groupby(xs)} if len(y)>0 else {i:np.array([]) for i in range(len(Xs))}
            for j in range(len(labels)):
                key = labels[j]
                if key in Xs:
                    self.__predict_base(Xs[key],ys[key],model=model['snode'][j])

    def predict(self,X,y=np.array([])):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)
        self._ypre = []
        self.__predict_base(X,y=y,model=self.model)
        return pd.concat(self._ypre).sort_index()

    '''post prune'''
    def __post_prune(self,model={}):
        if 'snode' in model and 'err' in model:
            subtrees = model['snode']
            errs, traversal = 0, False
            for j in range(len(subtrees)):
                if 'snode' in subtrees[j]:
                    traversal = True
                    break
                else:
                    errs += subtrees[j]['err'] if 'err' in subtrees[j] else 0
            if traversal:
                for j in range(len(subtrees)):
                    self.__post_prune(subtrees[j])
            else:
                errf = model['err']
                if errf <= errs:
                    model.pop('node')
                    model.pop('snode')
                    model.pop('con')

    def post_prune(self,X,y):
        err_b,err_a = 1,0
        count = 1
        while err_a < err_b:
            print('The %dth pruning.' % count)
            err_b = np.sum(self.predict(X,y) != y)
            self.__post_prune(model=self.model)
            err_a = np.sum(self.predict(X,y) != y)
            count += 1

##train a tree to fit the moons
#m = 1000
#X, y = sklearn.datasets.make_moons(m,noise=0.15)
#clf=decision_tree(10,split_rule='GINI')
#clf.fit(X,y)
#ypre=clf.predict(X)
#print(np.sum(y==ypre))
##plot_decision_boundary(X,y,feat_engi_func,clf.predict)
#m = 400
#Xt, yt = sklearn.datasets.make_moons(m,noise=0.15)
#ytpre=clf.predict(Xt,yt)
#print(np.sum(yt==ytpre))
#plot_decision_boundary(Xt,yt,feat_engi_func,clf.predict)
#
#clf.post_prune(Xt,yt)
#ytpre=clf.predict(Xt,yt)
#print(np.sum(yt==ytpre))
#plot_decision_boundary(Xt,yt,feat_engi_func,clf.predict)