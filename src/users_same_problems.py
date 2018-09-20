#!py2 or py3
#gives users with exactly k codes that have the same problems sets
#used by saveDataframe.py
import os

#make {user:['prob_id','prob_id',...} for all users in a directory
def get(mydir):
    user_file = {}
    for root, dir, file in os.walk(mydir, topdown = True):
        user = os.path.basename(root)
        subfile = [f[f.find("p")+1:f.find(".")] for f in file]
        user_file[user] = subfile
    #find the largest set of users with the same problem set
    visited = []
    user_max = [[],[]] #[list of users with same set of codes, list of the codes]
    for key1 in user_file.keys():
        user_temp = [key1]
        for key2 in user_file.keys():
            if key2 not in visited and not key1 == key2:
                #intersect = list(set.intersection(set(user_file[key1]) , set(user_file[key2])))
                if user_file[key1] == user_file[key2]:
                    user_temp.append(key2)
        visited.extend(user_temp)
        if len(user_temp)> len(user_max[0]):
            user_max[0] =  user_temp
            user_max[1] = user_file[key1]
    return(user_max)

if __name__ == '__main__':
    mydir = os.path.dirname(__file__)+ '/SourceCode_byYear_ordered/2012/4'
    user_max = get(mydir)
    print(user_max)
    print(len(user_max[0]))


