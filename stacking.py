#the goal of this project is to get better at python while also learning stacking
#and improve use of github

#DATASET CAN BE FOUND HERE
#https://archive.ics.uci.edu/ml/datasets/abalone

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#load dataset, prepare the dataset
df = pd.read_csv('abalone.data', header=None)

#split target and features, convery to numpy arrays
y = df[8].to_numpy()
X = df.drop([8], axis=1)
    
# one-hot encoding for feature 0
X = pd.concat([X.drop([0], axis=1), pd.get_dummies(X[0])], axis=1).to_numpy()
# 3 sexes, Male, Female, Infant

#normalize dataset to be positive values between 0 and 1
#decision to normalize to positive values only is because all attributes are positive measurements
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#normalize target variable(age)
#we shall be storing the min/max to rescale data later
min_y = min(y)
max_y = max(y)
y = (y - min_y) / (max_y - min_y)