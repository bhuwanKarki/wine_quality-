#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:59:21 2018

@author: bhuwankarki
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

red_df=pd.read_csv("/Users/bhuwankarki/Desktop/homework1/winequality-red.csv",sep=";")
white_df=pd.read_csv("/Users/bhuwankarki/Desktop/homework1/winequality-white.csv", sep=";")
color_red = np.repeat("red", red_df.shape[0])
color_white = np.repeat("white", white_df.shape[0])
red_df.insert(loc=0,column="type",value=color_red)
white_df.insert(loc=0,column="type",value=color_white)
wine_df=red_df.append(white_df)
replace_list = {"type" : {"red": 1, "white" : 0,}}
wine_df.replace(replace_list,inplace=True)

X=wine_df.drop(columns="type",axis=1).values
y=wine_df.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

alpha= list(np.linspace(0.01, 10,num=20))
alpha= list(np.linspace(0.01, 10,num=20))