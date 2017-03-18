#!/usr/bin/env python

import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import random

sys.path.append(os.path.join(os.environ['ROOTDIR'],'shared_tools'))
from logging_tools import Logger

logger = Logger()

def parse_csv(filename,n_samples=10000):

    import csv

    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        for row in datareader:
            if count > n_samples:
                return
            if count == 0:
                yield row
            elif random.random() < 0.05:
                yield row
            else:
                continue

            count += 1

    return

def reduce_csv(in_file,out_file='',n_samples=10000):

    if out_file == '':
        out_file = '%s_subset.csv' % in_file.split('.')[0]

    f_o = file(out_file,'w')               
    
    for row in parse_csv(in_file,n_samples):
        print type(row)
        print row
        f_o.write(row)

    f_o.close()

    return 1

def import_and_sample(filename,n_samples=10000):

    sample_size = len(pd.read_csv(filename,header=0,usecols=[0]))

    skipped_rows = list(set(range(1,sample_size))-set(random.sample(range(sample_size),n_samples)))
    

    df = pd.read_csv(filename,header=0,skiprows=skipped_rows)

    return df

def import_dataset(filename):

    df = pd.read_csv(filename,header=0)

    return df

def plot_bar(input_data,var_label=''):

    count_dict = {}
    for el in input_data:
        if el not in count_dict:
            count_dict[el] = 0
        count_dict[el] += 1

    labels = count_dict.keys()

    df = pd.DataFrame({'label':labels,'count':map(lambda x: count_dict[x],labels)},index=labels)
    df = df.sort('count',ascending=True)

    plt.figure()
    df['count'].plot(kind='barh')
    plt.title(var_label)
    plt.savefig('figures/plot_%s.pdf' % var_label)
    plt.close()

    return 1

def plot_all_variables(df):

    for var in df.columns:

        var_list = df[var].dropna()

        if len(var_list) < 1:
            logger.log('%s is null column. Continuing...' % var)
            continue
        
        tt = type(var_list.values[0])

        logger.log('Plotting %s (%s)' % (var,tt))        

        plt.figure()
        if 'float' in str(tt):
            var_list.plot(kind='hist',bins=10)
            plt.xlabel(var)
            plt.savefig('figures/plot_%s.pdf' % var)
            plt.close()            
        elif tt is str:
            if len(np.unique(var_list.values)) > 10:
                continue
            else:
                plot_bar(var_list,var)

    return 1

def plot_all_twoclass(df_class1,df_class2):    

    from plotting_tools import plot_multi_hist

    str_inputs = []

    for var in df_class1.columns:
        
        var_list1 = df_class1[var].dropna()
        var_list2 = df_class2[var].dropna()
        
        tt = type(var_list1.values[0])

        if 'float' not in str(tt):
            if tt is str:
                str_inputs.append(var)
            continue

        logger.log('Class sizes: %s (1), %s (2)' % (len(var_list1),len(var_list2)))
        logger.log('Class 1')
        logger.log('  Mean: %s' % round(np.mean(var_list1),1))
        logger.log('  Median: %s' % round(np.median(var_list1),1))
        logger.log('Class 2')
        logger.log('  Mean: %s' % round(np.mean(var_list2),1))
        logger.log('  Median: %s' % round(np.median(var_list2),1))
        logger.log(' ')

        high_bound = max(max(var_list1),max(var_list2))
        low_bound = min(min(var_list1),min(var_list2))        

        plot_multi_hist([var_list1,var_list2],10,[low_bound,high_bound],var,figure_name='figures/hist_%s' % var,group_labels=['Good','Bad'])
        
        
    return 1

def do_1D_plotting():
    
    #df_accepted = import_and_sample('accepted_2007_to_2016.csv')
    #df_rejected = import_and_sample('rejected_2007_to_2016.csv')

    df_accepted = import_dataset('accepted_2007_to_2016_subset400000.csv')
    
    logger.log('Number of accepted applicants in dataset: %s' % len(df_accepted))
    logger.log('Number of fields: %s ' % len(df_accepted.columns))

    plot_all_variables(df_accepted)

    return 1

def get_fields():

    #input_fields = map(lambda x: x.strip('\\n'),file('input_fields.csv').readlines())
    df = pd.read_csv('input_fields.csv',header=None,names=['field'])
    

    return list(df['field'])

def main():

    #df = pd.read_csv('accepted_2007_to_2016.csv',header=0)

    #df_paid = df[map(lambda x: 'Fully Paid' in x,df['loan_status'])]
    #df_not = df[map(lambda x: ('Charged Off' in x) or ('Late' in x) or x == 'Default',df['loan_status'])]

    import pickle
    #pickle.dump(df_paid,open('df_paid','wb'))
    #pickle.dump(df_not,open('df_notpaid','wb'))

    df_paid = pickle.load(file('df_paid_subset.p'))
    df_not = pickle.load(file('df_notpaid_subset.p'))

    input_fields = get_fields()
    
    df_paid = df_paid[input_fields]
    df_not = df_not[input_fields]

    plot_all_twoclass(df_paid,df_not)

    return 1

if __name__ == '__main__':
    main()

