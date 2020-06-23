#!py2
#used by saveDataframe.py
# AST feature extractor

import ast
import os
import ast_bigrams
import numpy
from sklearn.feature_selection import mutual_info_classif


#########################################################################################
#height of AST --> Proved not to be useful so we neglect them in our experiments.
#If you wish to include them in your feature dataframe, add them in savedataframe.py script.
#input: node, the result of  / output: height of each node +1
def ast_height(node):
    height = 1
    maxHeight = 0
    #for stat in tree.body:     d = 0
    if ast.iter_child_nodes(node):
        for child in ast.iter_child_nodes(node):
            childH = ast_height(child)
            if childH > maxHeight:
                maxHeight = childH
    return(height + maxHeight)


#make feature vector for feature labels
#input: directory of the dataset , mode = 'frequent' gives frequent bigrams otherwise all bigrams
def features(dataset_dir, mode):
    # features = ['height']
    features = []
    bigramset, bigramfreq = ASTbigrams.dataset_bigrams(dataset_dir)
    all_bigrams, frequent_bigrams, _ = ASTbigrams.bigrams_feature_vector(bigramset, 1)
    if mode == 'frequent':
        for bigram in frequent_bigrams:
            features.append(str(bigram))
        return(features , frequent_bigrams)
    else:
        for bigram in all_bigrams:
            features.append(str(bigram))
        return(features , all_bigrams)


#get features function
#input: address of .py file
#output: (1)data matrix in numpy(.training) --> normalized frequencies
#  (2)corresponding classes of data matrix in numpy(.target) (3)feature labels in numpy
def get_AST_features(dataset_dir, mode):
    feature_labels , dataset_bigrams = features(dataset_dir, mode = mode) #(1)height, bigrams, etc. or in str type(2)bigrams among all dataset
    classes = [] #users
    matrix_data = []
    for root, dir, file in os.walk(dataset_dir, topdown = True):
        user = os.path.basename(root)
        for code in file:
            lines = len(open(os.path.join(root, code)).readlines())
            code_featureVec = []
            classes.append(user)
            tree = ast.parse(open(os.path.join(root, code)).read())
            #get height
            # height = max([ast_height(child) for child in ast.iter_child_nodes(tree)])
            # code_featureVec.append(height)
            #get bigrams
            code_bigrams , code_bigrams_freq = ASTbigrams.bigram_freq(ASTbigrams.code_bigram(tree))
            for bigram in dataset_bigrams:
                if bigram in code_bigrams_freq.keys():
                    code_featureVec.append(float(code_bigrams_freq[bigram])/lines)
                else:
                    code_featureVec.append(float(0))
            #add to overall feature matrix
            matrix_data.append(code_featureVec)
    matrix_data = numpy.asarray(matrix_data)
    classes = numpy.asarray(classes)
    feature_labels = numpy.asarray(feature_labels)
    return(matrix_data , classes , feature_labels)


#Selecte features based on information gain criteria (mutual information)
#returns features with information above info_thr
#input: data_matrix, classes and features
#output: (1)selected features as a list (2)dictionary of selected features and their IG {feature name: IG}
#(3) all feature - IG pairs as a dictionary {feature name: IG}
def IG_selector(data_matrix , classes , feature_labels, threshold):
    info_thr = threshold
    IG_pairs = dict(zip(feature_labels, mutual_info_classif(data_matrix, classes , discrete_features=True)))
    IG_features = [k for k in IG_pairs.keys() if IG_pairs[k] > info_thr]
    IG_pairsSelected = {k: IG_pairs[k] for k in IG_features}
    return(IG_features , IG_pairsSelected, IG_pairs)



#Main body - gives a vector (a list) of AST features
if __name__ == "__main__":
    mydir = os.path.dirname(__file__) + '/SourceCode_byYear_ordered/2012/9'
    data , classes , features = get_AST_features(mydir, mode= 'all')
    IG_features , IG_pairsSelected , IG_pairs= IG_selector(data , classes , features, 0.9)
    # print(features)
    print('all features:' , len(features))
    print('IG selected:' , len(IG_features))
    print(type(IG_features))
    # for f in IG_features:
    #     print(f)


