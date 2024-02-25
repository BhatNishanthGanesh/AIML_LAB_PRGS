import pandas as pd

df=pd.read_csv('D:/cand.csv')
df=df.drop(['slno'],axis=1)
concepts=df.values[:,:-1]
target=df.values[:,-1]
df.head()

def learn(concepts,target):
    specific_h=concepts[0].copy()
    general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    for i,h in enumerate(concepts):
        if target[i]=="yes":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]='?'
                    specific_h[x]='?'
        if target[i]=="no":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
        indices=[i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]
    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    return specific_h,general_h

s_final,g_final=learn(concepts,target)
print(f"S_final:{s_final}")
print(f"G_final:{g_final}")
        