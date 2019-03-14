# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:21:37 2019

@author: duchezbr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#%%

train = pd.read_csv(r"C:\Users\kqvc199\Downloads\house-prices\train.csv")
test = pd.read_csv(r"C:\Users\kqvc199\Downloads\house-prices\test.csv")

df = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
#%% Only evaluate numerical data by removing all features with datatype 'object'

numerical_feats = df.dtypes[df.dtypes != 'object'].index
 
df[numerical_feats].info()

#%% Replace null values with median of the column
df[numerical_feats] = df[numerical_feats].fillna(df[numerical_feats].median())

#%%  Correlation matrix to see what features strongly correlate with the target variable

cor_matrix = pd.concat([df[:train.shape[0]][numerical_feats], train['SalePrice']], axis=1)
cor = cor_matrix.corr()

cor_target = abs(cor["SalePrice"])

#%% Split data for training purposes

X_train = df[:train.shape[0]][numerical_feats]
y = train['SalePrice'].values

#%% Train model iteratively changing the allowed correlation with the target variable
reg = LinearRegression()
cross_val_score(reg, X_train, y, cv=5).mean()

corr_coef = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
reg_score=list()
for i in corr_coef:
    relevant_features = cor_target[cor_target>i]
    
    if relevant_features.shape[0] < 2:
        break
    else:
        # remove sale price
        relevant_features = relevant_features.index[:-1]
        X_train = df[:train.shape[0]][relevant_features]
        reg_score.append(cross_val_score(reg, X_train, y, cv=5).mean())
        print("Coefficient: " + str(i) + "\nScore: " + str(cross_val_score(reg, X_train, y, cv=5).mean()))
        
reg_score = pd.Series(reg_score, index=corr_coef[0:len(reg_score)])
reg_score.plot()
plt.xlabel('correlation coefficient')
plt.ylabel('score')

#%% Choosing to make predictions using features that have correlation coefficient of 0.1
# get feature names
relevant_features = cor_target[cor_target>0.1]
# create model
X_train = df[:train.shape[0]][relevant_features.index[:-1]]
reg = LinearRegression()
reg.fit(X_train, y)
# test model
X_test = df[train.shape[0]:][relevant_features.index[:-1]]

predictions = reg.predict(X_test)
predictions 

submission = pd.DataFrame({'Id': test['Id'].values,'SalePrice': predictions})
#%%
submission.to_csv(r"C:\Users\kqvc199\Downloads\house-prices\20190314_house-prices_test.csv", index=False)
