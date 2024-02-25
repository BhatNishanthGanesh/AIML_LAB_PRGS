import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn import preprocessing 
from sklearn.cluster import KMeans 
from sklearn.datasets import load_iris 
from sklearn.mixture import GaussianMixture 

iris=load_iris()
df=pd.DataFrame(iris['data'],columns=iris['feature_names'])
df['target']=iris['target']

x=df.iloc[:,:-1]
y=df['target']

scaler=preprocessing.StandardScaler()
scaler.fit(x)
x_scaled_array=scaler.transform(x)
x_scaled=pd.DataFrame(x_scaled_array,columns=x.columns)

plt.figure(figsize=(14,7))
colormap=np.array(['red','green','blue'])

#Real plot 
plt.subplot(1,3,1)
plt.scatter(x_scaled['petal length (cm)'],x_scaled['petal width (cm)'],c=colormap[y],s=40)
plt.title('Real')


#K-plot
plt.subplot(1,3,2)
model=KMeans(n_clusters=3,random_state=0)
pred_y=model.fit_predict(x)
pred_y=np.choose(pred_y,[1,0,2]).astype(np.int64)
plt.scatter(x_scaled['petal length (cm)'],x_scaled['petal width (cm)'],c=colormap[pred_y],s=40)
plt.title('KMeans')


#GMM plot
plt.subplot(1,3,3)
gmm=GaussianMixture(n_components=3,max_iter=200)
y_cluster_gmm=gmm.fit_predict(x_scaled)
y_cluster_gmm=np.choose(y_cluster_gmm,[2,0,1]).astype(np.int64)
plt.scatter(x['petal length (cm)'],x['petal width (cm)'],c=colormap[y_cluster_gmm],s=40)
plt.title('GMM Classification')

plt.show()