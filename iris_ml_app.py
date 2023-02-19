import streamlit as st
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

lr = pickle.load(open('lr_model.pkl','rb'))
dt = pickle.load(open('dt_model.pkl','rb'))
rf = pickle.load(open('rf_model.pkl','rb'))
knn = pickle.load(open('knn_model.pkl','rb'))



st.header('Flower classification Prediction')

ml_model =[ 'LogisticRegression','RandomForestClassifier','KNeighborsClassifier',
            'DecisionTreeClassifier']

option = st.sidebar.selectbox('Select one of the ML model',ml_model)

sl = st.slider('Sepal Length',0.0,10.0)
sw = st.slider('Sepal Width',0.0,10.0)
pl = st.slider('Petal Length',0.0,10.0)
pw = st.slider('Petal Width',0.0,10.0)

test = [[sl,sw,pl,pw]]

st.write('Test Data',test)

if st.button('Run Classifier'):
    if option == 'LogisticRegression':
        st.success(lr.predict(test)[0])
    elif option == 'KNeighborsClassifier':
        st.success(knn.predict(test)[0])   
    elif option == 'DecisionTreeClassifier':
        st.success(dt.predict(test)[0])
    else:
        st.success(rf.predict(test)[0])
