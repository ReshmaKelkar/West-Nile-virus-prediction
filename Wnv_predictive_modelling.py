# WNV prediction -- Kaggle challenge(https://www.kaggle.com/c/predict-west-nile-virus)
# Objective of Challenge: Given weather, location, testing, and spraying data, this competition 
# asks you to predict when and where different species of mosquitoes will test positive for West Nile virus. 
# A more accurate method of predicting outbreaks of West Nile virus in mosquitoes will help the City of Chicago 
# and CPHD to more efficiently and effectively allocate resources towards preventing transmission of this potentially deadly virus. 
# Input data files are kept under the "input/" directory.
# Generated Comma separated output files in "input/" directory -- sampleSubmission.csv
# Certain parts of the code have been commented upon to aid analysis of the result. 
# Author - Reshma

import numpy as np 
import pandas as pd 
import os
from subprocess import check_output
from utils import *
from models import *

MISSING_VALUE = -999    

    
if __name__ == "__main__":
    import sys
    import time
    
    filename_prefix = 'sampleSubmission'

    if len(sys.argv)>1:
        print ('Usage: "python West_nile_virus_prediction.py"')
        exit()
        
    input_dir     = 'input/'
    print(check_output(["ls", input_dir]).decode("utf8"))
    
    #~ out_dir       = input_dir+'predictive/'
    #~ out_dir_plot  =  out_dir+'plot/'

    #~ if not os.path.exists(out_dir):
        #~ print 'Creating directory %s' % out_dir
        #~ os.mkdir(out_dir)
    
    #~ if not os.path.exists(out_dir_plot):
        #~ print 'Creating directory %s' % out_dir_plot
        #~ os.mkdir(out_dir_plot)

    # Load input datasets
    print "Reading input files"
    train, test, spray, weather,  mapdata = read_input_files(input_dir) 
    print "Training and test samples sizes are  %d, %d respectively" % (len(train), len(test))

    assert np.all(train != MISSING_VALUE) and np.all(test != MISSING_VALUE)
    weather = process_weather(weather)

    train, test = process_date(train), process_date(test)
    train, test = get_duplicated_rows(train), get_duplicated_rows(test)

    train = fix_leakage(train)
    
    train, test = get_closest_station(train), get_closest_station(test)
    train, test = merge_weather(train, weather), merge_weather(test, weather)
    
    test.replace(['UNSPECIFIED CULEX'], ['CULEX PIPIENS'], inplace=True) # replace Unspecified species with PIPIENS 
    train, test = get_dummies(train), get_dummies(test)
    
    drop_cols = ['Address', 'Block', 'Street', 'Trap', 'AddressNumberAndStreet', 'AddressAccuracy',
                 'Date', 'Species', 'Station']
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    train, test, y_reg, y_clf, test_id = prepare_data_for_model(train, test)
    
    regs = get_regressor_models()
    print "Running stack model"
    train_stack, test_stack = get_stacked_models(train, test, y_reg, regs)
    train, test  = train_stack, test_stack
    
    start_time = time.time()
    preds_xgb, imp_feature_xgb, fpr_xgb, tpr_xgb = model_XGB(train, test, y_clf)
    print "Time required to run/tune XGB:", time.time() - start_time
    
    #~ start_time = time.time()
    #~ preds_rf, imp_feature_rf, fpr_rf, tpr_rf = model_RF(train, test, y_clf, out_dir+filename_prefix)
    #~ print "Time required to run/tune RF:", time.time() - start_time
    
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    pd.DataFrame({"ID": test_id, "WnvPresent": preds_xgb}).to_csv(input_dir+filename_prefix+'.csv',index=False)
    
    #~ imp_feature_xgb.to_csv(out_dir+filename_prefix +'_'+timestamp+'_xgb_feature_importance.csv', sep='\t', encoding='utf-8')
    #~ pd.DataFrame({"ID": test_id, "WnvPresent": preds_rf}).to_csv(out_dir+filename_prefix +'_'+timestamp+'_rf.csv',index=False)
    #~ imp_feature_rf.to_csv(out_dir+filename_prefix +'_'+timestamp+'_rf_feature_importance.csv', sep='\t', encoding='utf-8')
    #~ plot_roc_auc(fpr_xgb, tpr_xgb, fpr_rf, tpr_rf, filename_prefix, out_dir_plot)



