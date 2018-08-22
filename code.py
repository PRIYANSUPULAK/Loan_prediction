#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 12:46:42 2018

@author: priyansu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("training_set.csv")
apple=dataset.describe()
dataset['Dependents'] = dataset['Dependents'].replace({'3+':'5'})
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset["Married"].mode()[0], inplace=True)
dataset["Dependents"].fillna(dataset["Dependents"].mode()[0],inplace=True)
dataset["Self_Employed"].fillna(dataset["Self_Employed"].mode()[0],inplace=True)
dataset["Loan_Amount_Term"].fillna(dataset["Loan_Amount_Term"].mode()[0],inplace=True)
dataset["Credit_History"].fillna(dataset["Credit_History"].mode()[0],inplace=True)
dataset["LoanAmount"].fillna(dataset["LoanAmount"].mean(),inplace=True)
x=dataset.iloc[:,1:12].values
y=dataset.iloc[:,12].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label=LabelEncoder()
label2=LabelEncoder()
label3=LabelEncoder()
label4=LabelEncoder()
label5=LabelEncoder()
label6=LabelEncoder()
x[:,0]=label.fit_transform(x[:,0])
x[:,1]=label2.fit_transform(x[:,1])
x[:,3]=label3.fit_transform(x[:,3])
x[:,4]=label4.fit_transform(x[:,4])
x[:,10]=label5.fit_transform(x[:,10])
one=OneHotEncoder(categorical_features=[0])
x=one.fit_transform(x).toarray()
y=label6.fit_transform(y)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=300,criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)


y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



