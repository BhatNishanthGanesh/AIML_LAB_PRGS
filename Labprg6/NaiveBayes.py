def probAttr(data,attr,val):
    Total=data.shape[0]
    cnt=len(data[data[attr]==val])
    return cnt,cnt/Total 

def train(data,attr,conceptVals,concept):
    conceptProb={}
    countConcept={}
    for cval in conceptVals:
        countConcept[cval],conceptProb[cval]=probAttr(data,concept,cval)
    attrConcept={}
    probability_list={}
    for att in attr:
        probability_list[att]={}
        attrConcept[att]={}
        for val in attr[att]:
            attrConcept[att][val]={}
            a,probability_list[att][val]=probAttr(data,att,val)
            for cval in conceptVals:
                dataTemp=data[data[att]==val]
                attrConcept[att][val][cval]=len(dataTemp[dataTemp[concept]==cval])/countConcept[cval]
        print(f"P(X):{conceptProb}\n")
        print(f"P(X|A):{attrConcept}\n")
        print(f"P(A):{probability_list}\n")
    return conceptProb,attrConcept,probability_list

def test(examples,attr,concept_list,conceptProb,attrConcept,probability_list):
    misclassification_count=0
    Total=len(examples)
    for ex in examples:
        px={}
        for a in attr:
            for x in ex:
                for c in concept_list:
                    if x in attrConcept[a]:
                        if c not in px:
                            px[c]=conceptProb[c]*attrConcept[a][x][c]/probability_list[a][x]
                        else:
                            px[c]=px[c]*attrConcept[a][x][c]/probability_list[a][x]
        print(px)
        classification=max(px,key=px.get)
        print(f"Classification:{classification} Expected: {ex[-1]}")
        if(classification!=ex[-1]):
            misclassification_count+=1
            misclassification_rate=misclassification_count*100/Total
            accuracy=100-misclassification_rate
            
    print(f"Misclassification Count: {misclassification_count}")
    print(f"Misclassification Rate: {misclassification_rate}")
    print(f"Accuarcy:{accuracy}%")


import pandas as pd

df=pd.read_csv("D:/id3.csv")
df.keys()[0]

concept=str(list(df)[-1])
concept_list=set(df[concept])
attr={}

for a in df.columns[:-1]:
    attr[a]=set(df[a])
    print(f"{a}:{attr[a]}")

conceptProb,attrConcept,probability_list=train(df,attr,concept_list,concept)
examples=pd.read_csv('D:/id3.csv')
test(examples.values,attr,concept_list,conceptProb,attrConcept,probability_list)