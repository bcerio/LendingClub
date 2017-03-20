#!/usr/bin/env python

import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')

from pandas import read_csv
import pickle

def main():

    df = read_csv('loan.csv',header=0)
    status_list = list(df['loan_status'].unique())

    print 'Number of loans in dataset: %s' % len(df)

    for ss in status_list:

        print 'Number of loans with status %s: %s' % (ss,len(df[df['loan_status'] == ss]))

    df_paid = df[map(lambda x: x == 'Fully Paid',df['loan_status'])]
    df_not = df[map(lambda x: x == 'Default' or x == 'Charged Off',df['loan_status'])]

    import pickle
    pickle.dump(df_paid,open('df_paid_subset.p','wb'))
    pickle.dump(df_not,open('df_notpaid_subset.p','wb'))

    return 1

if __name__ == '__main__':
    main()

