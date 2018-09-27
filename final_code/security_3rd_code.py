#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 21:31:49 2018

@author: deanng
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import time
from contextlib import contextmanager
from security_3rd_property import DATA_PATH, OVR_CLASS_NUM, NUM_ROUND, skf
from security_3rd_model import xgbMultiTrain

# TIME-COST FUNCTION
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f}s".format(title, time.time() - t0))
   
def main():
    with timer('LOAD FEATURE DATA'):
        # LOAD FEATURE V1
        train_1 = pd.read_csv(DATA_PATH+'/data/train_base_features_v1.csv') 
        test_1 = pd.read_csv(DATA_PATH+'/data/test_base_features_v1.csv')
        # LOAD FEATURE V2
        train_2 = pd.read_csv(DATA_PATH+'/data/train_base_features_v2.csv')
        test_2 = pd.read_csv(DATA_PATH+'/data/test_base_features_v2.csv')

        interaction_feat = train_2.columns[train_2.columns.isin(test_2.columns.values)].values
        train_2 = train_2[interaction_feat]
        test_2 = test_2[interaction_feat]
        # LOAD FEATURE V3
        train_3 = pd.read_csv(DATA_PATH+'/data/train_base_features_v3.csv')
        test_3 = pd.read_csv(DATA_PATH+'/data/test_base_features_v3.csv')
    
        interaction_feat = train_3.columns[train_3.columns.isin(test_3.columns.values)].values
        train_3 = train_3[interaction_feat]
        test_3 = test_3[interaction_feat]
      
        # MERGE ALL FEATURES
        train = train_1.merge(train_2, on=['file_id'], how='left')
        test = test_1.merge(test_2,on=['file_id'], how='left')
        train = train.merge(train_3, on=['file_id'], how='left')
        test = test.merge(test_3,on=['file_id'], how='left')
        print('[TRAIN SIZE]: ', train.shape)
        print('[TEST SIZE]: ', test.shape)
        
        # TRAIN DATA PREPARE
        X = train.drop(['file_id','label'], axis=1)
        y = train['label']
        print('[TRAIN FEATURE SIZE]: ', X.shape)
        print('[TRAIN LABEL DISTRIBUTION]: ')
        print(y.value_counts()) 

    with timer('ADD ONE_VS_REST PROB'):
        extra_feat_val = pd.read_csv(DATA_PATH + '/data/tr_lr_oof_prob.csv')
        extra_feat_test = pd.read_csv(DATA_PATH + '/data/te_lr_oof_prob.csv')
        prob_list = ['prob'+str(i) for i in range(OVR_CLASS_NUM)]
        X_extra = pd.concat(
            [X, extra_feat_val[prob_list]], axis=1)
        test_extra = pd.concat(
            [test, extra_feat_test[prob_list]],axis=1)


    #        X_extra = pd.concat([X, extra_feat_val[['prob0','prob1','prob2','prob3','prob4','prob5']]], axis=1)
#        test_extra = pd.concat([test, extra_feat_test[['prob0','prob1','prob2','prob3','prob4','prob5']]], axis=1)
    
    with timer('5-Fold Multi-Class Model Training'):

        # Variables
        logloss_rlt = []
        p_val_all = pd.DataFrame()
        p_test_all = pd.DataFrame(np.zeros((test.shape[0], 6)))
    
        # Start 5-fold CV
        for fold_i,(tr_index,val_index) in enumerate(skf.split(X, y)):
            print('FOLD -', fold_i, ' Start...')
            # Prepare train, val dataset
            X_train, X_val = X_extra.iloc[tr_index,:], X_extra.iloc[val_index,:]
            y_train, y_val = y[tr_index], y[val_index]
            # Train model
            model, p_val, p_test = xgbMultiTrain(X_train, X_val, y_train, y_val, test_extra, NUM_ROUND)
            # Evaluate Model and Concatenate Val-Prediction
            m_log_loss = log_loss(y_val, p_val)
            print('----------------log_loss : ', m_log_loss, ' ---------------------')
            logloss_rlt = logloss_rlt + [m_log_loss]
            truth_prob_df = pd.concat([y_val, p_val], axis=1)
            p_val_all = pd.concat([p_val_all, truth_prob_df], axis=0)
            # Predict Test Dataset
            p_test_all = p_test_all + 0.2*p_test
        
    with timer('Evaluation'):
        print('[LOGLOSS]: ', logloss_rlt)
        print('[LOGLOSS MEAN]: ', log_loss(p_val_all.iloc[:,0], p_val_all.iloc[:,1:]))
        #print('[LOGLOSS STD]: ', np.std(logloss_rlt))
        print('[LOGLOSS STD]: ', pd.Series(logloss_rlt).std())
        feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
        print('[TOP20 IMPORTANT FEATURES(5TH-FOLD MODEL)]: ')
        print(feat_imp[:20])
    
    with timer('SUBMIT CHECK'):
        rlt = pd.concat([test['file_id'], p_test_all], axis=1)    
        rlt.columns = ['file_id'] + prob_list
        check_flag = all(rlt.iloc[:,1:].sum(axis=1)-1<1e-6)
        if check_flag:
            print('RESULT IS OK...')
            rlt.to_csv(DATA_PATH+'/submit/rlt_TEST.csv', index=None)
            print('RESULT SAVED...')
        else:
            print('RESULT IS WRONG!')


if __name__ == '__main__':
    main()







