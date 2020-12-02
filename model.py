# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:40:58 2020

@author: Neetu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('C:/Users/PC/Desktop/fwdpricepredictionmodel/My football player price prediction.csv')
#df['fpl_sel'] = df['fpl_sel'].apply(lambda x: float(x[:-1])/100)

df1 = df.dropna()

##drop redundant and useless variables
new = df1[['market_value','fpl_value','fpl_sel','fpl_points','age_cat','big_club']]
new['fpl_sel'] = new['fpl_sel'].apply(lambda x: float(x[:-1])/100)

Y_encoded=new['market_value']
X_encoded=new.drop(['market_value'],axis=1)
###after modelling, final variables are in this df


from sklearn.model_selection import train_test_split
x_train_encoded,x_test_encoded,y_train_encoded,y_test_encoded=train_test_split(X_encoded,Y_encoded,test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
S_train_x_encoded=scaler2.fit_transform(x_train_encoded)

from sklearn.ensemble import RandomForestRegressor 
  
# create regressor object 
#regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor = RandomForestRegressor(bootstrap= False, max_depth= None, max_features=1, min_samples_split= 2, n_estimators= 413)
# fit the regressor with x and y data 
regressor.fit(S_train_x_encoded, y_train_encoded)   

# Saving model to disk
pickle.dump(regressor, open('C:\\Users\\PC\\Desktop\\fwdpricepredictionmodel\\model.pkl','wb'))
filehandler_st = open('C:\\Users\\PC\\Desktop\\fwdpricepredictionmodel\\standardized1.pkl','wb')
pickle.dump(scaler2,filehandler_st)
filehandler_st.close()
# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))