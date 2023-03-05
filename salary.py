import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_excel("data.xlsx")
print(df.head())
x=df.iloc[:,[0,1]].values
y=df.iloc[:,2].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#y_pred=regressor.predict(x_test)
#print(y_pred)
new_data=[2,6]
output=regressor.predict([new_data])
print(output)