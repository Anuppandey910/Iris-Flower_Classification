# import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
# # print(df)
# # print(df['petal_width'].value_counts())
# # print(df['petal_length'].value_counts())

# # print(df.isnull().sum())
# print(df.duplicated().sum())
# df.drop_duplicates(inplace=True)
# print(df.duplicated().sum())

# Split the data into dependent and independent variable
x = df.drop('label',axis=1)
y = df['label']
# print(type(x))
# print(type(y))

# Import library from sklearn module
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_split=12)
rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10,min_samples_split=12)
knn = KNeighborsClassifier(n_neighbors=8)


lr.fit(x_train,y_train)
dt.fit(x_train,y_train)
rf.fit(x_train,y_train)
knn.fit(x_train,y_train)

# print(lr.predict([[5,2,3,2]]))

# Save the model

# import pickle

# pickle.dump(lr,open('lr_model.pkl','wb'))
# pickle.dump(dt,open('dt_model.pkl','wb'))
# pickle.dump(rf,open('rf_model.pkl','wb'))
# pickle.dump(knn,open('knn_model.pkl','wb'))

