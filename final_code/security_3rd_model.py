#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:01:52 2018

@author: deanng
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from security_3rd_property import OVR_CLASS_NUM,CLASS_NUM, skf
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# MULTI-CALSS XGB PATAMETER
xgb_params_multi = {'objective':'multi:softprob',
              'num_class':CLASS_NUM,
              'eta': 0.04,  
              'max_depth':6,
              'subsample':0.9,
              'colsample_bytree':0.7,
              'lambda': 2,
              'alpha': 2,
              'gamma': 1,
              'scale_pos_weight':20,
              'eval_metric': 'mlogloss',
              'silent':0,
              'seed':149}

api_vec = TfidfVectorizer(ngram_range=(1,4),
                      min_df=3, max_df=0.9, 
                      strip_accents='unicode', 
                      use_idf=1,smooth_idf=1, sublinear_tf=1)

def tfidfModelTrain(train, test):
    tr_api = train.groupby('file_id')['api'].apply(lambda x:' '.join(x)).reset_index()
    te_api = test.groupby('file_id')['api'].apply(lambda x:' '.join(x)).reset_index()
    tr_api_vec = api_vec.fit_transform(tr_api['api'])
    val_api_vec = api_vec.transform(te_api['api'])
    return (tr_api_vec,val_api_vec)  


# NB-LR
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(x, y):
    y = y.values
    r = np.log(pr(x,1,y) / pr(x,0,y))
    np.random.seed(0)
    m = LogisticRegression(C=6, dual=True,random_state=0)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def nblrTrain(tr_tfidf_rlt, te_tfidf_rlt, train):
    label_fold=[]
    preds_fold_lr=[]
    lr_oof = pd.DataFrame()
    preds_te = []
    for fold_i,(tr_index,val_index) in enumerate(skf.split(train, train['label'])):
        if fold_i>=0:
            tr,val = train.iloc[tr_index],train.iloc[val_index]
            x = tr_tfidf_rlt[tr_index,:]
            test_x = tr_tfidf_rlt[val_index,:]        
            preds = np.zeros((len(val), OVR_CLASS_NUM))
            preds_te_i = np.zeros((te_tfidf_rlt.shape[0],OVR_CLASS_NUM))
            labels = [i for i in range(OVR_CLASS_NUM)]
            for i, j in enumerate(labels):
                print('fit', j)
                m,r = get_mdl(x, tr['label'] == j)
                preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
                preds_te_i[:,i] = m.predict_proba(te_tfidf_rlt.multiply(r))[:,1]
            preds_te.append(preds_te_i)
            preds_lr = preds
            lr_oof_i = pd.DataFrame({'file_id':val['file_id']})
            for i in range(OVR_CLASS_NUM):
                lr_oof_i['prob'+str(i)] = preds[:,i]
            lr_oof = pd.concat([lr_oof,lr_oof_i],axis=0)
    
            for i,j in enumerate(preds_lr):
                preds_lr[i] = j/sum(j)
            #log_loss_i = log_loss(val['label'], preds_lr)
            #print(log_loss_i)
            label_fold.append(val['label'].tolist())
            preds_fold_lr.append(preds_lr)
            
            lr_oof = lr_oof.sort_values('file_id')
            preds_te_avg = (np.sum(np.array(preds_te),axis=0) / 5)
            lr_oof_te = pd.DataFrame({'file_id':range(0,te_tfidf_rlt.shape[0])})
            for i in range(OVR_CLASS_NUM):
                lr_oof_te['prob'+str(i)] = preds_te_avg[:,i]
    return (lr_oof, lr_oof_te)

def xgbMultiTrain(X_train, X_val, y_train, y_val, test, num_round):

    # multi-cls model
    dtrain = xgb.DMatrix(X_train, y_train)      
    dval = xgb.DMatrix(X_val, y_val)    
    dtest = xgb.DMatrix(test.drop(['file_id'], axis=1))
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(xgb_params_multi,
                      dtrain, 
                      num_round, 
                      evals=watchlist, 
                      early_stopping_rounds=100,
                      verbose_eval=100
                     )
    p_val = pd.DataFrame(model.predict(dval, ntree_limit=model.best_iteration), index=X_val.index)  
    p_test = pd.DataFrame(model.predict(dtest, ntree_limit=model.best_iteration), index=test.index)
    return (model, p_val, p_test)

