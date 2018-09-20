#!py2
#Gives all bigrams of dataset
#output: (1) union of all bigrams used by each user (2) union of bigrams used by more than one user
#used by AST_features.py

import os
import ast
import collections

#get bigram of one source code
def code_bigram(node):
    result = []
    if node.__class__ == ast.Module:
        for child in ast.iter_child_nodes(node):
            bigram_collector = code_bigram(child)
            # print('after module:', bigram_collector)
            result = result + bigram_collector
            # print('after module result:', result)
            # print(len(result))
    else:
        for child in ast.iter_child_nodes(node):
            bigram= [ast.dump(node) , ast.dump(child)]
            result.append(bigram)
            bigram_collector = code_bigram(child)
            if not bigram_collector == []:
                result = result + bigram_collector
    return(result)

#remove duplicated in bigrams and add frequencies of each
#input: bigrams pair for a code (output of code_bigram())
# outputs: (1) set of tupled bigrams {(,),...) (2) dic(Counter) of frequency of tupled bigrams {(,):f,...}
def bigram_freq(bigrams):
    #bigram_freq_holder = {}
    bigram_ListofTuple = []
    for bigram in bigrams:
        bigram_ListofTuple.append(tuple(bigram))
    bigram_freq_holder = collections.Counter(bigram_ListofTuple)
    bigram_SetofTuple = set(bigram_ListofTuple)
    return(bigram_SetofTuple , bigram_freq_holder)

#Gives all bigrams of a dataset
#input: directory of the dataset
#output: (1) dic of users and their set of tupled bigrams (2) dic of users and dic of tupled bigrams and their freq
def dataset_bigrams(mydir):
    all_users_bigramset = {}
    all_users_bigramfreq = {}
    count = 0
    for root, dir, file in os.walk(mydir, topdown = True):
        if count>0: #to ignore the main directory
            user = os.path.basename(root)
            # get bigrams of each users codes -> union them -> add to dictionary of bigrams-by-user
            for code in file:
                try:
                    tree = ast.parse(open(os.path.join(root, code)).read())
                except:
                    print('code not compatible with python2 ast module')
                    continue
                code_bigrams, code_bigrams_freq = bigram_freq(code_bigram(tree))
                if user in all_users_bigramset:
                    all_users_bigramset[user] = set.union(all_users_bigramset[user] , code_bigrams)  #dic of set (of tuples)
                else:
                    all_users_bigramset[user] = code_bigrams
                if user in all_users_bigramfreq:
                    for key in code_bigrams_freq:
                        if key in all_users_bigramfreq[user]:
                            all_users_bigramfreq[user][key] += code_bigrams_freq[key]
                        else:
                            all_users_bigramfreq[user][key] = code_bigrams_freq[key]
                else:
                    all_users_bigramfreq[user] = code_bigrams_freq #dic of dic (of tuples as keys and freq as values)

        count = 1
    return(all_users_bigramset , all_users_bigramfreq)

#gives bigrams of a dataset as a feature vector
#input: dataset bigrams sets (first output of dataset_bigrams) and i = threshold for frequent bigrams
#output: feature vector as a list of tuples (1) all (2) frequent more than i times among users
# (3) dictionary of tuples and their respective freq, 1 (all bgr) is as its keys and their freq {(,):f}
def bigrams_feature_vector(bigramset, i):
    union = set()
    frequent = {} #{(,):freq among all users_at most once counted per user that has used a bgr k times}
    # all_union = [] #for union of all bigrams among users
    frequent_union = []  # for union of bigrams repeated more than i times among users
    for key in bigramset:
        union = set.union(bigramset[key], union)
    all_union = list(union)
    for key in bigramset:
        for bigram in bigramset[key]:
            try:
                frequent[bigram] += 1
            except:
                frequent[bigram] = 1
    for bigr in frequent.keys():
        if frequent[bigr]>i:
            frequent_union.append(bigr)
    return(all_union , frequent_union, frequent)


#dataframe of users vs. bigrams feature vectors
if __name__ == '__main__':
    mydir = os.path.dirname(__file__)+ '/SourceCode_byYear_ordered/2014/6'
    freq_thr = 1
    bigramset , bigramfreq = dataset_bigrams(mydir)
    all_bigrams , frequent_bigrams, frequent = bigrams_feature_vector(bigramset, freq_thr)
    # the non-frequent bigrams
    diff = [i for i in all_bigrams if i not in frequent_bigrams]

    print(len(all_bigrams) , len(frequent_bigrams))
    print(len(diff))










