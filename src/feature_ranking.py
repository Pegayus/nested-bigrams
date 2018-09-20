#get a dataframe and rank features
#output: (1) inside ig_tuner and corr_tuner, save [#selected features,acc] in each step 
#             and print the whole list of x and list of y
# (2) save top features in a csv file with parent node in bgr in left col and the child in right col and importance score in third col
# (3) make dataframe with new features (modify the input dataframe and select the cols that are in the list of
#     selected bgrs. save the datafram.
import os
import pandas as pd
import operator
import numpy as np
import AST_features
from pandas import DataFrame as df
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif

#input: 1)numpy 2d array 2)list of calsses 3)feature vector(list of str) 4)hyperparams
#output: 1)list of scores of cv 2)avg of the list
def RFC(data, classes, IG_features, cv, ntree, maxdepth, crit):
    RF_classifier = RandomForestClassifier(n_estimators=ntree, criterion=crit,
                                           max_depth=maxdepth, oob_score=True)

    RF_classifier.fit(data, classes)
    tree_heights = [estimator.tree_.max_depth for estimator in RF_classifier.estimators_]
    treeHmin = min(tree_heights)
    treeHmax = max(tree_heights)
    treeHavg = sum(tree_heights) / len(tree_heights)
    print 'height of trees'
    print 'min = ', treeHmin, 'max = ', treeHmax, 'avg = ', treeHavg
    scores_IGFeatures = cross_val_score(RF_classifier, data, classes, cv=cv)
    accavg = sum(scores_IGFeatures) / len(scores_IGFeatures)
    print 'average acc = ', accavg
    return (scores_IGFeatures , accavg)


#input: 1){bgr:IG} all bgrs' igs in dic for a dataset 2)threshold for ig
#output: 1)list of bgr (str) selected by ig thr 2)dic of those selected features and their ig
def IG_selector(IG_pairs, threshold):
    info_thr = threshold
    IG_features = [k for k in IG_pairs.keys() if IG_pairs[k] > info_thr]
    IG_pairsSelected = {k: IG_pairs[k] for k in IG_features}
    return(IG_features , IG_pairsSelected)

#get the least number of features that give acc above 85% by tuning ig threshold
#input: 1)dir of dataset for feature extraction 2)ig thrs to test 3)hyperparams
#output: 1)2d numpy of codes vs features selected by tuner 2)classes 3)selected features as list of str
# 4)optimum ig thr 5) corresponding acc with that ig thr and selected features
def IG_tuner(data , classes , feature_labels, ig_thr_range, acc_thr, cv, ntree, maxdepth, crit):
    print 'IG_tuner activated.'
    print 'number of all features:', len(feature_labels)
    print('calculating IGs...')
    # dic of all features with their ig
    IG_pairs = dict(zip(feature_labels, mutual_info_classif(data, classes, discrete_features=True)))
    print('IG tuning started...')
    final_thr = 0
    final_features = feature_labels
    final_data = data
    final_acc = 0
    Xig = []
    Yig = []
    for thr in ig_thr_range:
        print 'ig_thr = ' , thr , '...'
        #list of ig features(list of strings), dic of selected ig features with their ig
        IG_features, IG_pairsSelected= IG_selector(IG_pairs, thr)
        print '# selected IG features = ', len(IG_features)
        if len(IG_features) == 0:
            break
        data_IG = []
        for feature_vec in data:
            feature_freq = dict(zip(feature_labels, feature_vec))
            temp = [feature_freq[f] for f in feature_freq.keys() if f in IG_features]
            data_IG.append(temp)
        #numpy 2d array, list of user's name for each row of data,list of selected features as string
        scores_IG , accavg = RFC(data_IG, classes, IG_features , cv, ntree, maxdepth, crit)
        Xig.append(len(IG_features))
        Yig.append(accavg)
        if accavg > acc_thr:
            if len(IG_features)< len(final_features):
                final_thr = thr
                final_features = IG_features
                final_data = data_IG
                final_acc = accavg
            continue
        # print 'selection terminated.'
        # break
    return(Xig , Yig , final_data, final_features, final_thr, final_acc)


#rank the selected features that give acc above 80% by one out approach
#input: numpy 2d, classes , list of features , hyperparams
#output: 1) ranks = [(bgr , imposcore)] 2) ranks_dic = {bgr: (rank , importance acc)}
def Ranker_oneOut(ig_data, classes, ig_features, cv, ntree, maxdepth, crit):
    #one-out selection
    ImpScores = {}
    count = 0
    for ftr in ig_features:
        count += 1
        print 'feature ' , count, ' processing...'
        temp_data = []
        temp_features = [i for i in ig_features if not i == ftr]
        for feature_vec in ig_data:
            feature_freq = dict(zip(ig_features, feature_vec))
            temp = [feature_freq[f] for f in feature_freq.keys() if f in temp_features]
            temp_data.append(temp)
        temp_acc , temp_accavg= RFC(temp_data, classes, temp_features, cv, ntree, maxdepth, crit)
        imp_score = 1 - temp_accavg
        ImpScores[ftr] = imp_score
    # print(ImpScores)
    ranks = sorted(ImpScores.iteritems(), key=operator.itemgetter(1), reverse=True)
    ranks_dic = {}
    R = 0
    for item in ranks:
        R += 1
        ranks_dic[item[0]] = (R , item[1])
    # ranks = [(bgr , rank)] , ranks_dic = {bgr: (rank , importance acc)}
    return(ranks , ranks_dic)


#input: (1)numpy 2d array for correlation matrix (2)list of feature labels 3)correlation thr
#output: list of selected features (list of str)
def Corr_selector(corr_matrix, feature_laebles, thr):
    corr = np.asarray(corr_matrix)
    row = range(len(corr))
    col = range(len(corr))
    out = []
    for i in row:
        if feature_laebles[i] not in out:
            temp = [x for x in col if x>i and feature_laebles[x] not in out]
            for j in temp:
                if abs(corr[i][j]) > thr:
                    out.append(feature_laebles[j])
    selected = [f for f in feature_laebles if f not in out]
    return(selected)

#get the least number of features that give acc above 85% by tuning correlation threshold
#input: 1)dir of dataset for feature extraction 2)corr thrs to test 3)hyperparams
#output: 1)2d numpy of codes vs features selected by tuner 2)classes 3)selected features as list of str
# 4)optimum corr thr 5) corresponding acc with that corr thr and selected features
def Corr_tuner(data , classes , feature_labels,  corr_thr_range, acc_thr, cv, ntree, maxdepth, crit):
    print 'corr tuner activated.'
    print 'number of all features:', len(feature_labels)
    print('calculating correlation matrix...')
    #matrix of correlation as a numpy 2d array
    corr_matrix = np.corrcoef(data, rowvar= False) #to consider columns as varibales
    print('correlation tuning started...')
    final_thr = 0
    final_features = feature_labels
    final_data = data
    final_acc = 0
    Xcor = []
    Ycor = []
    for thr in corr_thr_range:
        print 'corr_thr = ', thr, '...'
        # list of ig features(list of strings), dic of selected ig features with their ig
        corr_features = Corr_selector(corr_matrix, feature_labels, thr)
        print '# selected uncorrelated features = ', len(corr_features)
        if len(corr_features) == 0:
            break
        data_corr = []
        for feature_vec in data:
            feature_freq = dict(zip(feature_labels, feature_vec))
            temp = [feature_freq[f] for f in feature_freq.keys() if f in corr_features]
            data_corr.append(temp)
        # numpy 2d array, list of user's name for each row of data,list of selected features as string
        scores_corr, accavg = RFC(data_corr, classes, corr_features, cv, ntree, maxdepth, crit)
        Xcor.append(len(corr_features))
        Ycor.append(accavg)
        if accavg > acc_thr:
            if len(corr_features)< len(final_features):
                final_thr = thr
                final_features = corr_features
                final_data = data_corr
                final_acc = accavg
            continue
        # print 'selection terminated.'
        # break
    return (Xcor, Ycor, final_data, final_features, final_thr, final_acc)

if __name__ == '__main__':
    year = 'all'
    codes = '9'
    users = '81'
    # mode = '_frequent1sameprob_'
    mode = '_frequent1diffProb_'
    mydir = os.path.dirname(__file__) + '/dataframe/COPYuserWith9codes_729codeall9freqbgrmorethan1user.csv'
    # subdir = '/df_' + year + '_' + codes + mode + users + '.csv'
    # mydir = os.path.dirname(__file__) + '/dataframe' + subdir
    csv_saveto = os.path.dirname(__file__) + '/FeaturesRanking'
    df_saveto = os.path.dirname(__file__) + '/dataframe/ranking'
    if not os.path.exists(csv_saveto):
        os.makedirs(csv_saveto)
    if not os.path.exists(df_saveto):
        os.makedirs(df_saveto)
    ########################################################################################
    # #read dataframe
    # datadf = pd.read_csv(mydir)
    # data = datadf.drop(['classes', 'Unnamed: 0'], axis=1).values
    # classes = datadf['classes'].values
    # feature_labels = list(datadf.drop(['classes', 'Unnamed: 0'], axis=1))
    #
    # # hyperparameters
    # cv = int(codes)
    # ntree = 300
    # maxdepth = None
    # crit = 'entropy'
    # ig_thr_range = [0.1, 0.5, 0.9, 1, 1.2, 1.5, 2, 2.5, 3]
    # # ig_thr_range = [1, 1.2, 1.5, 2, 2.5, 3]
    # corr_thr_range = [0.9, 0.7, 0.5, 0.3, 0.1]
    # acc_thr = 0.90
    # ###########################################################################################
    ##################################  INFORMATION GAIN RANKING   ###############################
    # #give dataframe  infor and get (1)csv points (2)new df
    # Xig , Yig , ig_data , ig_features, igthr, igacc = IG_tuner(data , classes , feature_labels, ig_thr_range, acc_thr,
    #                                    cv, ntree, maxdepth, crit)
    # print('')
    # print 'final ig features: ', len(ig_features)
    # print 'thr = ' , igthr, 'avg_acc = ' , igacc
    # print ''
    #
    # #output (1)save dataframe with new features
    # print 'saving dataframe'
    # ig_newdatadf = pd.DataFrame(data= ig_data, columns= ig_features)
    # ig_newdatadf['classes'] = classes
    # print 'number of rows(codes)= ', len(ig_newdatadf)
    # print 'number of columns(features)= ', len(list(ig_newdatadf))
    # # save to csv
    # ig_newdatadf.to_csv(os.path.join(df_saveto, 'COPYIGdf_' + year + '_' + codes + '_frequent1sameprob_' + users + '_acc'+str(acc_thr)+'.csv'))

    # #ranking
    # print 'ranking starts...'
    # #1) ranks = sorted by score:[(bgr , imposcore)] 2) ranks_dic = {bgr: (rank , importance acc)}
    # # Ranker(ig_data, classes, ig_features, cv, ntree, maxdepth, crit)
    # ranks , ranks_dic = Ranker_oneOut(ig_data, classes, ig_features, cv, ntree, maxdepth, crit)
    #
    # #output (2)
    # csvfile = []
    # csvname = ['Node1' , 'Node2' , 'Score']
    # for item in ranks:
    #     try:
    #         item0 = eval(item[0])
    #     except:
    #         continue
    # #     item0 = eval(item[0])
    #     csvfile.append([item0[0],item0[1],item[1]])
    # csvdf = pd.DataFrame(data= csvfile, columns= csvname)
    # csvdf.to_csv(os.path.join(csv_saveto, 'COPYIGranks_' + year + '_' + codes + '_frequent1sameprob_' + users + '_acc'+str(acc_thr)+ '.csv'))

    #output (3) : x , y
    # print 'x_ig= ' , Xig
    # print 'y_ig= ' , Yig
    # #######################################################################################
    ########################  CORRELATION RANKING  #########################################
    # Xcor, Ycor, corr_data, corr_features, corr_thr, avg_acc = Corr_tuner(data, classes, feature_labels,
    #                                                                    corr_thr_range, acc_thr, cv, ntree, maxdepth, crit)
    # print('')
    # print 'final corr features: ', len(corr_features)
    # print 'thr = ', corr_thr, 'avg_acc = ', avg_acc
    # print ''
    #
    # # output (1)save dataframe with new features
    # print 'saving dataframe'
    # cor_newdatadf = pd.DataFrame(data=corr_data, columns=corr_features)
    # cor_newdatadf['classes'] = classes
    # print 'number of rows(codes)= ', len(cor_newdatadf)
    # print 'number of columns(features)= ', len(list(cor_newdatadf))
    # # save to csv
    # cor_newdatadf.to_csv(os.path.join(df_saveto, 'CORRdf_' + year + '_' + codes + '_frequent1sameprob_' + users + '_acc'+str(acc_thr)+ '.csv'))
    # # ranking
    # print 'ranking starts...'
    # # 1) ranks = sorted by score:[(bgr , imposcore)] 2) ranks_dic = {bgr: (rank , importance acc)}
    # # Ranker(ig_data, classes, ig_features, cv, ntree, maxdepth, crit)
    # ranks, ranks_dic = Ranker_oneOut(corr_data, classes, corr_features, cv, ntree, maxdepth, crit)
    # # ranks2, ranks_dic2 = Ranker_allOut(ig_data, classes, ig_features, cv, ntree, maxdepth, crit)
    #
    # # output (2)
    # csvfile = []
    # csvname = ['Node1', 'Node2', 'Score']
    # for item in ranks:
    #     try:
    #         item0 = eval(item[0])
    #     except:
    #         continue
    #         #     item0 = eval(item[0])
    #     csvfile.append([item0[0], item0[1], item[1]])
    # csvdf = pd.DataFrame(data=csvfile, columns=csvname)
    # csvdf.to_csv(os.path.join(csv_saveto, 'CORRranks_' + year + '_' + codes + '_frequent1sameprob_' + users + '_acc'+str(acc_thr)+ '.csv'))
    #
    #
    #
    #
    #
    # #summary
    # print('')
    # print 'final ig features: ', len(ig_features)
    # print 'thr = ', igthr, 'avg_acc = ', igacc
    #
    # print('')
    # print 'final corr features: ', len(corr_features)
    # print 'thr = ', corr_thr, 'avg_acc = ', avg_acc
    # print ''
    # # output (3) : x , y
    # print 'x_ig= ', Xig
    # print 'y_ig= ', Yig
    # # output (3) : x , y
    # print 'x_cor= ', Xcor
    # print 'y_cor= ', Ycor
    #
    # print ''
    # print 'acc_thr= ' , acc_thr

    ##########################################################
    # read dataframe
    mydir2 = os.path.dirname(__file__) + '/dataframe/userWith9codes_729codeall9freqbgrmorethan1user.csv'
    datadf = pd.read_csv(mydir)
    datadf2 = pd.read_csv(mydir2)
    data = datadf.drop(['classes', 'Unnamed: 0'], axis=1).values
    classes = datadf['classes'].values
    feature_labels = list(datadf.drop(['classes', 'Unnamed: 0'], axis=1))
    data2= datadf2.drop(['classes', 'Unnamed: 0'], axis=1).values
    classes2 = datadf2['classes'].values
    feature_labels2 = list(datadf2.drop(['classes', 'Unnamed: 0'], axis=1))


    # hyperparameters
    cv = int(codes)
    ntree = 300
    maxdepth = None
    crit = 'entropy'
    ig_thr_range = [1, 1.2, 1.5, 2, 2.5, 3]
    # ig_thr_range = [1, 1.2, 1.5, 2, 2.5, 3]
    corr_thr_range = [0.9, 0.7, 0.5, 0.3, 0.1]
    acc_thr = 0.90
    ###########################################################################################33
    # give dataframe  infor and get (1)csv points (2)new df
    Xig, Yig, ig_data, ig_features, igthr, igacc = IG_tuner(data, classes, feature_labels, ig_thr_range, acc_thr,
                                                            cv, ntree, maxdepth, crit)
    print('')
    print 'final ig features: ', len(ig_features)
    print 'thr = ', igthr, 'avg_acc = ', igacc
    print ''

    # output (1)save dataframe with new features
    finaldata=pd.DataFrame()
    print 'saving dataframe'
    for col in datadf2:
        if col in ig_features:
            finaldata = finaldata.append(datadf2[col])
    finaldata['classes'] = datadf2['classes']
    finaldata = finaldata.T

    # ig_newdatadf = pd.DataFrame(data=ig_data, columns=ig_features)
    # ig_newdatadf['classes'] = classes
    print 'number of rows(codes)= ', len(finaldata)
    print 'number of columns(features)= ', len(list(finaldata))
    # save to csv
    finaldata.to_csv(os.path.join(df_saveto,
                                     'COPYIGdf_' + year + '_' + codes + '_frequent1sameprob_' + users + '_acc' + str(
                                         acc_thr) + '.csv'))
