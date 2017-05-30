# This script produces classification model of XGBoost and RandomForest of pre-processed train data
# Used GridSearchCV to finetune parameters for both models
# Time taken to tune parameters by random forest ~ 10 min, Time taken to tune parameters by xgboost ~ 45 min
# Generated output files with information such as predicted probability, roc_auc curve, tuned parameter dictionary


import numpy as np
import pandas as pd
import xgboost as xgb
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso


def get_feature_imporance(train, model):
    '''
    Get coefficients of all features used in model
    '''
    feature_importances = zip(train.columns, model.feature_importances_)
    feature_importances = pd.DataFrame(feature_importances, columns=['feature', 'importance'])
    feature_importances = feature_importances.sort('importance', ascending=False)
    
    return feature_importances

def report(grid_scores, n_top=15):
    '''
    Write parameters 
    '''
    print ("The following are the  top %d parameters ......" % n_top)
    param_dict = {}
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
        if i == 0:
            param_dict=score.parameters
            
    return param_dict


def get_grid_search_model(model, parameters, train, target):
    '''
    Using GridSearchCV algorithm to fine tune parameters
    '''
    n_jobs = -1
    clf    = GridSearchCV(model, parameters, n_jobs=n_jobs, 
                         cv=StratifiedKFold(target, n_folds=5, shuffle=True),
                         scoring='roc_auc', verbose=0, refit=True)

    clf.fit(train, target)
    
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
   
    return report(clf.grid_scores_)
 
    
def get_roc_curve(X, y, model):
    '''
    Plot roc-auc curve on each cross validated model
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    
    fpr, tpr, threshold = roc_curve(y_test, preds)
    
    return fpr, tpr
    
    
def get_data_of_top_features(train, test, target, model):
    
    model.fit(train, target)
    good_features = train.columns.values[np.where(model.feature_importances_ > 0.005)]
    train         = train[good_features]
    test          = test[good_features]
    return train, test

    
def build_model(model, train, test, target):
    
    fpr, tpr = get_roc_curve(train, target, model)
    model.fit(train, target)
    test_probs  = model.predict_proba(test)[:,1]
    imp_feature = get_feature_imporance(train, model)
    
    return test_probs, imp_feature, fpr, tpr
    

    
def model_XGB(train, test, target, path=''):
    
    xgb_model  = xgb.XGBClassifier()
    parameters = {'nthread'         : [6], 
                  'objective'       : ['binary:logistic'],
                  'learning_rate'   : [0.05, 0.025],
                  'gamma'           : [1.5], 
                  'max_depth'       : [9, 11],
                  'min_child_weight': [9, 11],
                  'max_delta_step'  : [0],
                  'silent'          : [0, 1],
                  'subsample'       : [0.5, 0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators'    : [800, 900, 1000],
                  'missing'         : [-999],
                  'seed'            : [2017]
                  }
    
    #~ parameters = get_grid_search_model(xgb_model, parameters, train, target)
    #~ json.dump(parameters, open(path+"_tuned_parameter_xgb",'w'))
    
    #~ xgb_model  = xgb.XGBClassifier( n_estimators     = parameters["n_estimators"],
                                    #~ objective        = parameters["objective"],
                                    #~ learning_rate    = parameters["learning_rate"],
                                    #~ gamma            = parameters["gamma"],
                                    #~ max_depth        = parameters["max_depth"],
                                    #~ min_child_weight = parameters["min_child_weight"],
                                    #~ max_delta_step   = parameters["max_delta_step"],
                                    #~ silent           = parameters["silent"],
                                    #~ subsample        = parameters["subsample"],
                                    #~ colsample_bytree = parameters["colsample_bytree"],
                                    #~ missing          = parameters["missing"],
                                    #~ seed             = parameters["seed"]
                                  #~ )
                                  
    xgb_model  = xgb.XGBClassifier( n_estimators     = 900,
                                    objective        = 'binary:logistic',
                                    learning_rate    = 0.025,
                                    gamma            = 1.5,
                                    max_depth        = 11,
                                    min_child_weight = 9,
                                    max_delta_step   = 0,
                                    silent           = 0,
                                    subsample        = 0.5,
                                    colsample_bytree = 0.7,
                                    missing          = -999,
                                    seed             = 2017
                              )   # tuned parameters
                
                    
    #~ train, test = get_data_of_top_features(train, test, target,  xgb_model)
    test_probs, imp_feature, fpr, tpr = build_model(xgb_model, train, test, target)

    return test_probs, imp_feature, fpr, tpr


def model_RF(train, test, target, path=''):

    rf_model   = RandomForestClassifier()
    parameters = {'n_estimators'     : [800, 900, 1000],
                  'criterion'        : ['entropy'],
                  'max_depth'        : [10],
                  'min_samples_leaf' : [1, 20],
                  'max_features'     : [1, 20],
                  'bootstrap'        : [True, False],
                  'random_state'     : [2017]
                  }
    #~ parameters = get_grid_search_model(rf_model, parameters, train, target)
    #~ json.dump(d, open(path+"_tuned_parameter_rf",'w'))
    
    rf_model   = RandomForestClassifier( n_estimators     = parameters["n_estimators"],
                                         min_samples_leaf = parameters["min_samples_leaf"],
                                         bootstrap        = parameters["bootstrap"],
                                         criterion        = parameters["criterion"],
                                         max_features     = parameters["max_features"],
                                         max_depth        = parameters["max_depth"],
                                         random_state     = parameters["random_state"]
                                  )
    #~ train, test = get_data_of_top_features(train, test, target,  rf_model)                              
    test_probs, imp_feature, fpr, tpr = build_model(rf_model, train, test, target)
    return test_probs, imp_feature, fpr, tpr
 
 
def get_regressor_models():
    
    reg1 = RandomForestRegressor(n_estimators=750, criterion='mse', 
                                 max_depth=9, min_samples_split=2, 
                                 min_samples_leaf=6, min_weight_fraction_leaf=0.0,
                                 max_features=0.75, max_leaf_nodes=None, 
                                 bootstrap=False, oob_score=False,
                                 n_jobs=2, random_state=101, verbose=0, 
                                 warm_start=False)
                            
    reg2 = Ridge(               alpha=3.0, fit_intercept=True, normalize=False, 
                                copy_X=True, max_iter=None, tol=0.001, solver='auto') 
                
    reg3 = Lasso(               alpha=0.2, fit_intercept=True, normalize=False, 
                                precompute=False, copy_X=True, max_iter=1000, 
                                tol=0.0001, warm_start=False, positive=False, 
                                random_state=101, selection='cyclic')

    regs = [reg1, reg2, reg3]
    
    return regs
    
    
def get_stacked_models(train, test, y, regs, n=1): 
    '''
    train(pd data frame), test(pd date frame), Target data, List of clfs 
    to stack, position of last non-scailed model in clfs. get_stacked_models() 
    performs Stacked Aggregation on data: it uses 4 different models to 
    get out-of-fold predictions of log(number of mosquitos) for train data. 
    It uses the whole train dataset to obtain predictions for test. 
    This procedure adds 4 meta-features (predictions of 4 models) to both 
    train and test data.
    '''
    training    = train.as_matrix()
    testing     = test.as_matrix()
    train_cols  = train.columns.values
    test_cols   = test.columns.values
    blend_train = np.zeros((training.shape[0], len(regs))) 
    blend_test  = np.zeros((testing.shape[0], len(regs)))
    years       = np.unique(train.year.values) 
    
    for j, reg in enumerate(regs):
        print ('Training regressor [%s]' % (j))
        for i in range(len(years)):
            print ('Fold [%s]' % (i))
            X_tr       = training[train.year.values!=years[i]]
            Y_tr       = y[train.year.values!=years[i]]
            X_cv       = training[train.year.values==years[i]]
            scaler     = StandardScaler().fit(X_tr)                  
            X_tr_scale = scaler.transform(X_tr)
            X_cv_scale = scaler.transform(X_cv)
   
            X_tr = training[train.year.values!=years[i]]
            Y_tr = y[train.year.values!=years[i]]
            X_cv = training[train.year.values==years[i]]
            scaler=StandardScaler().fit(X_tr)   # scale data for linear models                          
            X_tr_scale=scaler.transform(X_tr)
            X_cv_scale=scaler.transform(X_cv)

            if j<n: # these models do not require scaling                                           
                reg.fit(X_tr, Y_tr)
                blend_train[train.year.values==years[i], j] = reg.predict(X_cv)
            else:    # these models DO require scaling 
                reg.fit(X_tr_scale, Y_tr)
                blend_train[train.year.values==years[i], j] = reg.predict(X_cv_scale)
                
        scaler=StandardScaler().fit(training)
        X_train_scale=scaler.transform(training)
        X_test_scale=scaler.transform(testing)   
        if j<n:                                             
            reg.fit(training, y)
            blend_test[:, j] = reg.predict(testing)
        else:
            reg.fit(X_train_scale, y)
            blend_test[:, j] = clf.predict(X_test_scale)
                
    new_cols_train = train_cols.tolist()+['reg1', 'reg2', 'reg3']
    new_cols_test  = test_cols.tolist()+['reg1', 'reg2', 'reg3']

    X_train_blend_full = pd.DataFrame(np.concatenate((training, blend_train), axis=1), columns=new_cols_train)
    X_test_blend_full  = pd.DataFrame(np.concatenate((testing, blend_test), axis=1), columns=new_cols_test)
       
    return X_train_blend_full, X_test_blend_full 
    

def plot_roc_auc(fpr_xgb, tpr_xgb, fpr_rf, tpr_rf, filename, path):
    '''
    Compare AUCs of XGB and RF model
    '''
    plt.figure(0).clf()
    roc_auc_x = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb,tpr_xgb, color='blue', label="Model XGB, auc="+str(roc_auc_x))
    roc_auc_r = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, color='darkorange',label="Model RF, auc="+str(roc_auc_r))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    plt.savefig(path+filename+'_'+'ROC_'+timestamp+'.png')
    plt.show()
    plt.clf()
