import pandas as pd
from pandas import DataFrame
from collections import Counter
from math import log 
from pprint import pprint

df_tennis=pd.read_csv('D:/id3.csv')
df_tennis.keys()[0]

def entropy(probs):
    return sum([-prob*log(prob,2) for prob in probs])

def entropy_of_list(a_list):
    cnt=Counter(x for x in a_list)
    num_instances=len(a_list)*1.0
    probs=[x/num_instances for x in cnt.values()]
    return entropy(probs)

def information_gain(df,split_attribute_name,target_attribute_name):
    df_split=df.groupby(split_attribute_name)
    nobs=len(df.index)*1.0
    df_agg_ent=df_split.agg({target_attribute_name:[entropy_of_list,lambda x:len(x)/nobs]})[target_attribute_name]
    df_agg_ent.columns=['Entropy','PropObservations']
    new_entropy=sum(df_agg_ent['Entropy']*df_agg_ent['PropObservations'])
    old_entropy=entropy_of_list(df[target_attribute_name])
    return old_entropy-new_entropy

def id3(df,target_attribute_name,attribute_name,default_class=None):
    cnt=Counter(x for x in df[target_attribute_name])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or (not attribute_name):
        return default_class
    else:
        default_class=max(cnt.keys())
        gain=[information_gain(df,attr,target_attribute_name) for attr in attribute_name]
        index_of_max=gain.index(max(gain))
        best_attr=attribute_name[index_of_max]
        tree={best_attr:{}}
        remaining_attribute_names=[i for i in attribute_name if i!=best_attr]

        for attr_val,datasubsets in df.groupby(best_attr):
            subtree=id3(datasubsets,target_attribute_name,remaining_attribute_names,default_class)
            tree[best_attr][attr_val]=subtree
    return tree 

attribute_name=list(df_tennis.columns)
attribute_name.remove("PlayTennis")
tree=id3(df_tennis,'PlayTennis',attribute_name)
print("\n\n The Resultant tree is \n")
pprint(tree)
          