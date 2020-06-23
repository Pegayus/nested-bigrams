#!py2
#take dir of data and save the dataframe. you can use sameprob function to save only the ppl who have the same
#  set of problems.

import ast_features
import os
from shutil import copy
import users_same_problems
from pandas import DataFrame as df

def sameProb(mydir, subdir):
    common_usersProbs = usersWithSameProbs.get(mydir)
    print '#users with same problem set = ', len(common_usersProbs[0])
    print ''
    if not os.path.exists(os.path.dirname(__file__) + '/SameProbs' + subdir):
        os.makedirs(os.path.dirname(__file__) + '/SameProbs' + subdir)
    for root, dir, file in os.walk(mydir, topdown=True):
        user = os.path.basename(root)
        if user in common_usersProbs[0]:
            dest = os.path.dirname(__file__) + '/SameProbs' + subdir + '/' + user
            if not os.path.exists(dest):
                os.mkdir(dest)
            files_dir = [os.path.join(root, f) for f in file]
            for src in files_dir:
                copy(src, dest)
    new_mydir = os.path.dirname(__file__) + '/SameProbs' + subdir
    return new_mydir

if __name__ == '__main__':
    year = '2014'
    codes = '4'
    num = '229' #number of users
    subdir = '/' + year + '/' + codes
    mydir = os.path.dirname(__file__) + '/SourceCode_byYear_ordered' + subdir
    newdir = sameProb(mydir , subdir)
    dest = os.path.dirname(__file__) + '/dataframe'
    if not os.path.exists(dest):
        os.mkdir(dest)
    print 'getting data, classes, feature_labels...'
    data, classes, feature_labels = AST_features.get_AST_features(newdir, mode='frequent')
    print 'making dataframe...'
    data_df = df(data= data, columns= feature_labels)
    data_df['classes'] = classes
    print 'number of rows(codes)= ', len(data_df)
    print 'number of columns(features)= ', len(list(data_df))
    # save to csv
    data_df.to_csv(os.path.join(dest, 'df_' + year + '_' + codes + '_frequent1diffProb_' + num + '.csv'))
    
