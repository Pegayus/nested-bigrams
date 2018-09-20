#get dataframe and implement RFC

#!py2 because it calls modules that need to run in py2
#RFC with features from one dataset and test on another datasets
#Classification based on ensemble random forest using scikit learn
#Implement k-fold cross validation and report average accuracies


import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import cross_validate



#for all users in the dataset
def RFC_all(data, classes, cv, ntrees, maxdepth, crit):
    RF_classifier = RandomForestClassifier(n_estimators= ntrees, criterion= crit,
                                           max_depth = maxdepth, oob_score= True)
    RF_classifier.fit(data, classes)
    tree_heights = [estimator.tree_.max_depth for estimator in RF_classifier.estimators_]
    treeHmin = min(tree_heights)
    treeHmax = max(tree_heights)
    treeHavg = sum(tree_heights)/len(tree_heights)
    # print 'height of trees'
    # print 'min = ' , treeHmin , 'max = ' , treeHmax, 'avg = ' , treeHavg
    scores_allFeatures = cross_val_score(RF_classifier, data, classes, cv = cv)
    # scores1 = cross_validate(RF_classifier, data, classes, cv=cv, return_train_score=True)
    return(scores_allFeatures)
    # return(scores_allFeatures, scores1)



if __name__ == '__main__':
    ###############################################################################
    Fsubdir = '/dataframe/df_2012_4_frequent1sameProb_70.csv'
    Fdir = os.path.dirname(__file__) + Fsubdir
    ###############################################################################
    #table
    # read dataframe
    datadf = pd.read_csv(Fdir)
    data = datadf.drop(['classes','Unnamed: 0'] , axis= 1).values
    classes = datadf['classes'].values
    feature_labels = list(datadf.drop(['classes', 'Unnamed: 0'] , axis= 1))
    print ''
    print '# all frequent features = ', len(feature_labels)

    print('RFC on data starting...')

    cv = 4 # = number of codes per user
    ntree = 300
    maxdepth = None
    crit = 'entropy'

    scores_all = RFC_all(data, classes, cv, ntree, maxdepth, crit)
    avg_acc= np.asarray(scores_all).mean()
    sd_acc= np.asarray(scores_all).std()
    # print('All-USERS with all features:')
    print(scores_all)
    print 'average=', avg_acc
    print 'standard deviation= ', sd_acc
    print ''
    # print 'another cv with training and testing scores:'
    # print scores1
