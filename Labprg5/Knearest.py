from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris=datasets.load_iris()
print("Iris dataset is loaded...")

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.1)

for i in range(len(iris.target_names)):
    print("Label: ",i," - ",str(iris.target_names[i]))

classifier=KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print("Classification of Kneighbors using knn with k=1")
for r in range(len(x_test)):
    print("Label: ",str(x_test[r])," Actual Label: ",str(y_test[r]))
    "Predicted Label: ",str(y_pred[r])
    
print("Classification Accuarcy: ",classifier.score(x_test,y_test))