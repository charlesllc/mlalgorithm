# -*- coding=utf-8 -*- 

import numpy as np
from math import log
from pandas.api.types import is_string_dtype

class decision_tree(object):
    def __init__(self,max_depth=5,split_rule='ID3'):
        self.max_depth = max_depth
        self.split_rule = split_rule
        self.model = {}

    '''compute entropy of samples'''
    def entropy(self,y):
        #whether samples are weighted according to y's type: True-weighted, False-not weighted
        no_weight = isinstance(y, pd.Series)
        if no_weight:
            p = y.groupby(y).count() / len(y)
            if self.split_rule=='GINI':
                pure = np.sum(p * (1-p))
            else:
                pure = np.sum(-p * np.log(p)) / log(2)
            return pure
        else:
            #y is a DataFrame with columns ['label','weight']
            p = y['weight'].groupby(y['label']).sum() / np.sum(y['weight'])
            if self.split_rule=='GINI':
                pure = np.sum(p * (1-p))
            else:
                pure = np.sum(-p * np.log(p)) / log(2)
            return pure
    '''compute entropy of the amount of labels or the probablity of labels'''
    def entropy_num(self,ynum):
        p = ynum.div(ynum.sum(axis=1), axis=0)
        if self.split_rule=='GINI':
            pure = np.sum(p*(1-p),axis=1)
        else:
            pure = np.sum(-p*np.log(p),axis=1) / log(2)
        return pure

    '''compute the conditional entropy'''
    def con_entropy(self,x,y):
        #True-weighted y, False-not weighted y
        no_weight = isinstance(y,pd.Series)
        #if a string variable, compute directly
        if is_string_dtype(x):
            if no_weight:
                tmp = y.groupby(x).agg(['count',self.entropy])
            else:
                tmp = pd.DataFrame()
                tmp['count'] = y['weight'].groupby(x).sum()
                tmp['entropy'] = pd.Series([it for _,it in y.groupby(x)]).map(self.entropy)
            prior_prob = tmp['count'] / tmp['count'].sum()
            post_ent = tmp['entropy']
            con_ent = np.sum(prior_prob*post_ent)
            cutoff = list(prior_prob.index)
            prior_prob.index = list(range(len(prior_prob)))
        #if a continuous variable, optimize the split point
        else:
            x.name = 1
            ptil = np.percentile(x,range(5,100,5))
            cutoffs = [-np.inf]+ sorted(set(list(ptil))) + [np.inf]
            labels = np.array([i for i in range(len(cutoffs)-1)])
            xs = pd.cut(x,cutoffs,labels=labels)
            post_cnt = y.groupby([y,xs]).count().unstack(level=0) if no_weight else\
                        y['weight'].groupby([y['label'],xs]).sum().unstack(level=0)
            post_cnt.fillna(0,inplace=True)
            post_cnt['prior'] = post_cnt.sum(axis=1)
            post_cnt_cum = post_cnt.cumsum()
            post_cnt_cum_op = post_cnt_cum.iloc[-1]-post_cnt_cum
            prior_P0 = post_cnt_cum.iloc[0:-1,-1]/post_cnt_cum.iloc[-1,-1]
            prior_P1 = post_cnt_cum_op.iloc[0:-1,-1]/post_cnt_cum.iloc[-1,-1]
            post_ent0 = self.entropy_num(post_cnt_cum.iloc[0:-1,0:-1])
            post_ent1 = self.entropy_num(post_cnt_cum_op.iloc[0:-1,0:-1])
            con_ents = prior_P0.mul(post_ent0,axis=0) + prior_P1.mul(post_ent1,axis=0)
            #choose the optimal threshold
            k = list(con_ents.index[con_ents==con_ents.min()])[0]
            con_ent,cutoff = con_ents.loc[k], cutoffs[k+1]
            prior_prob = pd.Series([prior_P0.loc[k],prior_P1.loc[k]])
        return con_ent,round(cutoff,6),prior_prob

    '''generate the split node'''
    def split_node(self,X,y):
        m,n = X.shape
        no_weight = isinstance(y,pd.Series)
        #compute entropy or gini
        if no_weight:
            res = X.apply(lambda x: self.con_entropy(x,y))
            ent = self.entropy(y)
        else:
            #choose the variables with non nan samples
            nmrate = 1 - X.isna().sum()/m
            nmrate = nmrate[nmrate>0]
            nmk = list(nmrate.index)
            res = pd.Series(nmk,index=nmk)
            ent = pd.Series(nmk,index=nmk)
            for k in nmk:
                x = X[k].dropna()
                y1 = y.loc[x.index]
                res.loc[k] = self.con_entropy(x,y1)
                ent.loc[k] = self.entropy(y1)
        #select the optimal split node
        pures = pd.Series([v[0] for v in res], index=res.index)
        if self.split_rule=='GINI':
            pures = pures if no_weight else pures/nmrate
            k = pures.index[pures==pures.min()][0]
        else:
            gain = ent-pures if no_weight else nmrate*(ent-pures)
            if self.split_rule=='C4.5':
                prior_prob = pd.DataFrame([v[2] for v in res], index=res.index)
                ivalue = self.entropy_num(prior_prob)
                gain_ratio = gain.div(ivalue, axis=0)
                k = gain_ratio.index[gain_ratio==gain_ratio.max()][0]
            else:
                k = gain.index[gain==gain.max()][0]
        return k, res.loc[k][1]

    '''train decision tree by recursion and save tree as a json string in the case of no weights'''
    def __fit_base(self,X,y,model={},depth=0):
        #predict y in the current node when it is a leaf.
        ycnt = y.groupby(y).count()
        ypre = list(ycnt[ycnt==ycnt.max()].index)
        model['ypre'] = ypre[0]
        #split down
        if depth < self.max_depth-1 and len(ycnt) > 1:
            split,cutoff = self.split_node(X,y)
            model['node'] = split
            is_string = isinstance(cutoff, list)
            xs= X[split] if is_string else np.sign(np.sign(X[split]-cutoff)-1)+1
            Xs = [item for _,item in X.groupby(xs)]
            ys = [item for _,item in y.groupby(xs)]
            model['con'] = {}
            model['con']['is_str'] = is_string
            model['con']['bins'] = cutoff if is_string else [-np.inf, cutoff, np.inf]
            model['snode'] = [{} for v in range(len(ys))]
            for l in range(len(ys)):
                self.__fit_base(Xs[l],ys[l], model=model['snode'][l], depth=depth+1)

    '''train decision tree by recursion and save tree as a json string in the case of weights'''
    def __fit_base_na(self,X,Y,model={},depth=0):
        y,w = Y['label'], Y['weight']
        #predict y in the current node like a leaf.
        ycnt = w.groupby(y).sum()
        ypre = list(ycnt[ycnt==ycnt.max()].index)
        model['ypre'] = ypre[0]
        allna= ((~X.isna()).apply(lambda x: y[x].nunique())>1).sum()
        #split down waiting until all conditions satisfy.
        if depth < self.max_depth-1 and len(ycnt) > 1 and allna>0 and len(y)>3:
            #generate the split feature and split point
            split,cutoff = self.split_node(X,Y)
            is_string = isinstance(cutoff, list)
            #generate the groupby labels.
            xs= X[split].copy(deep=True) if is_string else np.sign(np.sign(X[split]-cutoff)-1)+1
            if xs.isna().sum()>0:
                xsna,yna,Xna,wna = xs[xs.isna()], y[xs.isna()], X[xs.isna()], w[xs.isna()]
                cutlist = cutoff if is_string else [0,1]
                mna,nbin = len(xsna),len(cutlist)
                index=sorted(list(xsna.index)*nbin)
                yna = yna.repeat(nbin)
                xsna = pd.Series(cutlist*mna, index=index)
                xs.dropna(inplace=True)
                prior_prob = list(w.groupby(xs).sum()/w.sum())
                wna = pd.Series(prior_prob*mna, index=index).mul(wna)
                Xna = Xna.apply(lambda x:x.repeat(nbin), axis=0)
                y,w,X = y.loc[xs.index], w.loc[xs.index], X.loc[xs.index]
                y,w,X,xs = y.append(yna), w.append(wna), pd.concat([X,Xna]), xs.append(xsna)
                Y = pd.DataFrame({'label':y, 'weight':w})
            #groupby X and Y
            Xs = [item for _,item in X.groupby(xs)]
            Ys = [item for _,item in Y.groupby(xs)]
            #save the model
            model['node'] = split
            model['con'] = {}
            model['con']['is_str'] = is_string
            model['con']['bins'] = cutoff if is_string else [-np.inf, cutoff, np.inf]
            model['snode'] = [{} for v in range(len(Ys))]
            #recurse the above steps
            for l in range(len(Ys)):
                self.__fit_base_na(Xs[l],Ys[l], model=model['snode'][l], depth=depth+1)
    '''fit function'''
    def fit(self,X,y):
        m,n=X.shape
        #convert into DataFrame or Series
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)
        else:
            self.columns_o = {k:v for v,k in zip(X.columns,range(n))}
            X.columns = list(range(n))
        #choose a fit_base function according to the number of nan samples
        if X.isna().sum().sum()>0:
            Y = pd.DataFrame({'label':y, 'weight':np.ones(m)})
            self.__fit_base_na(X,Y,model=self.model,depth=0)
        else:
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
            for j in range(len(label)):
                key = label[j]
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

#self = decision_tree(max_depth=5,split_rule='C4.5')