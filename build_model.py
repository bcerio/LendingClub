#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')

import sys
import os
import random
from operator import itemgetter

reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

from sklearn import cross_validation
from sklearn import grid_search
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn import decomposition

sys.path.append(os.path.join(os.environ['ROOTDIR'],'shared_tools'))
from logging_tools import Logger


logger = Logger()

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokenizer = RegexpTokenizer(r'\w+')
        word_vec = tokenizer.tokenize(doc)
        return [self.wnl.lemmatize(t) for i,t in enumerate(word_vec)]

def one_hot_encode(input_data):

    input_list = []

    for i in input_data:
        temp_dict = {i:1}
        input_list.append(temp_dict)
    
    dv = feature_extraction.DictVectorizer(sparse=False)
    output_data = dv.fit_transform(input_list)

    return output_data,dv.vocabulary_

def plot_roc_curve(y_test,probs):

    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_test,probs)
    roc_auc = auc(fpr, tpr)

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.pdf')
    plt.close()

    return 1

def clean_emp_length(df_column):

    emp_length_list = []
    for ss in df_column.values:

        if ss == 'n/a':
            new_length = 0
        else:            
            new_length = int(ss.rstrip('+ years').lstrip('< '))

        emp_length_list.append(new_length)

    return emp_length_list        

def clean_emp_title(df_column):

    out_list = []

    for tt in df_column.values:

        if tt is None or tt == '' or tt == 'nan' or tt == 'NaN':
            out_list.append(0)
        else:
            out_list.append(1)

    return out_list

def get_region(df_column):

    out_list = []
    
    northeast = ['CT','NY','PA','NJ','OH','RI','MA','MD','VT','DC','NH','DE','ME']

    south = ['GA','NC','TX','VA','MO','FL','KY','SC','LA','AL','KS','WV','MS','TN','AR']

    midwest = ['IL','MN','WI','KS','MI','SD', 'MT','WY','OK','IA','NE','ID','IN','ND']

    west = ['AZ','CA','OR','UT','WA','CO','NV','AK','NM','HI']

    for s in df_column.values:

        if s in northeast:
            out_list.append('NORTHEAST')
        elif s in south:
            out_list.append('SOUTH')
        elif s in midwest:
            out_list.append('MIDWEST')
        elif s in west:
            out_list.append('WEST')
        else:
            out_list.append('OTHER')

    return out_list

def build_tfidf(df_column,do_svd=True,n_components=50):

    import nltk

    df_column = df_column.fillna(value='')
    desc_cleaned = []
    
    for dd in df_column.values:
        if 'Borrower added' in dd:
            dd_new = '>'.join(dd.split('>')[1:])
        else:
            dd_new = ''
        dd_new = dd_new.replace('<br>',' ')
        desc_cleaned.append(dd_new)

    stop_words = nltk.corpus.stopwords.words('english')
    stop_words += map(str,range(0,20))
    stop_words += map(str,range(1900,2020))
    stop_words += map(lambda x: '0%s' % x,range(1,10))
    stop_words += ['00','000']

    tf_idf = feature_extraction.text.TfidfVectorizer(stop_words=stop_words,
                                                     ngram_range=(1,2),
                                                     min_df=3,
                                                     max_df=0.50,
                                                     norm='l2',
                                                     tokenizer=LemmaTokenizer(),
                                                     max_features=1000 if do_svd else n_components)

    X_desc = tf_idf.fit_transform(desc_cleaned)
    X_desc = X_desc.toarray()

    if do_svd:
        svd = decomposition.TruncatedSVD(n_components=n_components,random_state=44)
        X_desc = svd.fit_transform(X_desc)

    return X_desc,tf_idf.vocabulary_

def get_credit_history_len(issue_date,earliest_date):

    out_list = []

    import datetime    

    for i in xrange(len(issue_date)):

        id = datetime.datetime.strptime(issue_date[i],'%b-%Y')
        ed = datetime.datetime.strptime(earliest_date[i],'%b-%Y')
        diff = id - ed
        out_list.append(float(diff.days)/365)

    return out_list

def get_dataset():

    import pickle

    derived_var_list = ['credit_length','loan_amt_to_income','loan_amt_to_income_joint','region']

    df_paid = pickle.load(file('df_paid_subset.p'))
    df_paid['is_default_or_late'] = [0]*len(df_paid)
    df_not = pickle.load(file('df_notpaid_subset.p'))
    df_not['is_default_or_late'] = [1]*len(df_not)

    df_paid = df_paid.sample(n=len(df_not))
    df = pd.concat([df_paid,df_not])

    ex_instance = df_paid.values[0]
    for i in xrange(len(df_paid.columns)):
        print df_paid.columns[i],ex_instance[i]

    df['emp_length'] = clean_emp_length(df['emp_length'])
    df['emp_title'] = clean_emp_title(df['emp_title'])

    df['credit_length'] = get_credit_history_len(df['issue_d'].values,df['earliest_cr_line'].values)

    df['loan_amt_to_income'] = df['loan_amnt']/df['annual_inc']
    df['loan_amt_to_income_joint'] = df['loan_amnt']/df['annual_inc_joint']

    df['region'] = get_region(df['addr_state'])

    df['application_type'] = map(lambda x: int(x == 'INDIVIDUAL'),df['application_type'])

    df['verification_status'] = map(lambda x: int(x == 'Verified' or x == 'Source Verified'),df['verification_status'])
    
    from eda import get_fields
    input_fields = get_fields()
    input_fields += derived_var_list
    
    good_fields = []
    for col in df.columns:

        if not ('float' in str(df[col].dtype) or 'int' in str(df[col].dtype)):
            continue

        #df_temp = df_paid[col].dropna()
        #if float(len(df_temp))/len(df_paid) < 0.15:
        #    print 'removing %s since %s of data is missing from field' % (col,1-float(len(df_temp))/len(df_paid))
        #    continue
        
        good_fields.append(col)

    input_fields = list(set(input_fields) & set(good_fields))

    #X_title,title_to_indy = one_hot_encode(df['emp_title'].values)
    X_reason,reason_to_indy = one_hot_encode(df['purpose'].values)
    X_location,state_to_indy = one_hot_encode(df['region'].values)
    X_home,home_to_indy = one_hot_encode(df['home_ownership'].values)

    nlp_svd = True
    X_desc,desc_vocab = build_tfidf(df['desc'],nlp_svd,25)

    additional_fields = map(lambda x: x[0],sorted(map(lambda x: (x,reason_to_indy[x]),reason_to_indy.keys()),key=itemgetter(1)))
    additional_fields += map(lambda x: x[0],sorted(map(lambda x: (x,state_to_indy[x]),state_to_indy.keys()),key=itemgetter(1)))
    additional_fields += map(lambda x: x[0],sorted(map(lambda x: (x,home_to_indy[x]),home_to_indy.keys()),key=itemgetter(1)))
    if not nlp_svd:
        additional_fields += map(lambda x: 'DESC_%s' % x[0],sorted(map(lambda x: (x,desc_vocab[x]),desc_vocab.keys()),key=itemgetter(1)))
    else:
        additional_fields = map(lambda x: 'DESC_%s' % x,range(X_desc.shape[1]))

    y = df['is_default_or_late'].values
    df = df[input_fields].fillna(value=-1)
    
    X = df.values
    X = np.concatenate((X,X_reason),axis=1)
    X = np.concatenate((X,X_location),axis=1)
    X = np.concatenate((X,X_home),axis=1)
    X = np.concatenate((X,X_desc),axis=1)
    
    input_fields = input_fields + additional_fields

    return X,y,input_fields

def main(do_svd=False):

    X,y,headers = get_dataset()

    if do_svd:

        svd = decomposition.TruncatedSVD(n_components=15,random_state=44)
        X = svd.fit_transform(X)
    
    #imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
    #imp = imp.fit(X)

    #X = imp.transform(X)
    
    # split sample into random subsets for training and testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.35,random_state=42)

    logger.log('Shape of training sample: %s' % str(X_train.shape))
    logger.log('Shape of test sample: %s' % str(X_test.shape))

    classifier = ensemble.RandomForestClassifier(verbose=0)
    
    parameters = {'n_estimators':[50,100,200,500],'max_depth':[5,10,None],'criterion':['entropy']}
    opt_classifier = grid_search.GridSearchCV(classifier,parameters,verbose=1,n_jobs=8,scoring='recall')

    opt_classifier.fit(X_train,y_train)

    logger.log('Optimal hyperparameters: %s' % opt_classifier.best_params_)
    logger.log('Best score: %s' % opt_classifier.best_score_)
    logger.log('Score on hold-out: %s' % opt_classifier.score(X_test,y_test))
    logger.log('Accuracy score on hold-out: %s' % metrics.accuracy_score(y_test,opt_classifier.best_estimator_.predict(X_test)))    

    #for i in range(10):
    #    rand_indy = random.randint(0,X_test.shape[0]-1)
    #    logger.log('Predicted class: %s' % opt_classifier.predict([X_test[rand_indy]]))
    #    logger.log('Actual class: %s' % y_test[rand_indy])

    probs = opt_classifier.predict_proba(X_test)[:,1]
    plot_roc_curve(y_test,probs)
        
    logger.log('Feature importance')
    logger.log(' ')
    fi = sorted([(headers[i],imp) for i,imp in enumerate(opt_classifier.best_estimator_.feature_importances_)],key=itemgetter(1),reverse=True)
    for feat_imp in fi:
        logger.log('%s: %s' % feat_imp)
        
    return 1

if __name__ == '__main__':
    main()

