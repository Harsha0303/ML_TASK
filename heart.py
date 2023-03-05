import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("heart.csv")
#print(df.head())
x=df.iloc[:,[0,1]].values
y=df.iloc[:,2].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
dtc1=DecisionTreeClassifier(random_state=0)
dtc1.fit(x_train,y_train)
y_pred1=dtc1.predict(x_test)
classifier_knn=KNeighborsClassifier(n_neighbors=4)
classifier_knn.fit(x_train,y_train)
y_pred2=classifier_knn.predict(x_test)
print(y_pred1)
print(y_pred2)