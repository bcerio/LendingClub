#!/usr/bin/env python

import sys
import os
import random
from operator import itemgetter

reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn import grid_search
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_extraction

sys.path.append(os.path.join(os.environ['ROOTDIR'],'shared_tools'))
from logging_tools import Logger


logger = Logger()

def one_hot_encode(input_data):

    input_list = []

    for i in input_data:
        temp_dict = {i:1}
        input_list.append(temp_dict)
    
    dv = feature_extraction.DictVectorizer(sparse=False)
    output_data = dv.fit_transform(input_list)

    return output_data,dv.vocabulary_

def get_dataset():

    import pickle

    df_paid = pickle.load(file('df_paid_subset.p'))
    df_paid['is_default_or_late'] = [0]*len(df_paid)
    df_not = pickle.load(file('df_notpaid_subset.p'))
    df_not['is_default_or_late'] = [1]*len(df_not)

    from eda import get_fields
    input_fields = get_fields()    
    
    good_fields = []
    for col in df_paid.columns:

        if not ('float' in str(df_paid[col].dtype) or 'int' in str(df_paid[col].dtype)):
            continue

        #df_temp = df_paid[col].dropna()

        #if len(df_temp) == 0:
        #    continue
        
        good_fields.append(col)

    input_fields = list(set(input_fields) & set(good_fields))

    df_paid = df_paid.sample(n=len(df_not))

    df = pd.concat([df_paid,df_not])

    #X_title,title_to_indy = one_hot_encode(df['emp_title'].values)
    X_reason,reason_to_indy = one_hot_encode(df['purpose'].values)

    print reason_to_indy
    additional_fields = map(lambda x: x[0],sorted(map(lambda x: (x,reason_to_indy[x]),reason_to_indy.keys()),key=itemgetter(1)))
    print additional_fields
    print X_reason.shape

    X = df[input_fields].values
    X = np.concatenate((X,X_reason),axis=1)
    input_fields = input_fields + additional_fields
    y = df['is_default_or_late'].values

    return X,y,input_fields

def main():

    X,y,headers = get_dataset()

    imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
    imp = imp.fit(X)

    X = imp.transform(X)
    
    # split sample into random subsets for training and testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.75,random_state=42)

    logger.log('Shape of training sample: %s' % str(X_train.shape))
    logger.log('Shape of test sample: %s' % str(X_test.shape))

    classifier = ensemble.RandomForestClassifier(verbose=0)
    
    parameters = {'n_estimators':[50,100,250],'max_depth':[2,5,None],'criterion':['gini','entropy']}
    opt_classifier = grid_search.GridSearchCV(classifier,parameters,verbose=1,n_jobs=8,scoring='recall')

    opt_classifier.fit(X_train,y_train)

    logger.log('Optimal hyperparameters: %s' % opt_classifier.best_params_)
    logger.log('Best score: %s' % opt_classifier.best_score_)
    logger.log('Score on hold-out: %s' % opt_classifier.score(X_test,y_test))
    logger.log('Accuracy score on hold-out: %s' % metrics.accuracy_score(y_test,opt_classifier.best_estimator_.predict(X_test)))    

    for i in range(10):
        rand_indy = random.randint(0,X_test.shape[0]-1)
        logger.log('Predicted class: %s' % opt_classifier.predict([X_test[rand_indy]]))
        logger.log('Actual class: %s' % y_test[rand_indy])

    logger.log('Feature importance')
    logger.log(' ')
    fi = sorted([(headers[i],imp) for i,imp in enumerate(opt_classifier.best_estimator_.feature_importances_)],key=itemgetter(1),reverse=True)
    for feat_imp in fi:
        logger.log('%s: %s' % feat_imp)
        
    return 1

if __name__ == '__main__':
    main()

