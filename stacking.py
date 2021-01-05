#the goal of this project is to get better at python while also learning stacking
#and improve use of github, spyder and other programs

#%% load dataset
#DATASET CAN BE FOUND HERE
#https://archive.ics.uci.edu/ml/datasets/abalone
import pandas as pd
df = pd.read_csv('abalone.data', header=None)

#%% prepare the dataset
#split target and features, convery to numpy arrays
y = df[8].to_numpy()
X = df.drop([8], axis=1)
    
# one-hot encoding for feature 0
X = pd.concat([X.drop([0], axis=1), pd.get_dummies(X[0])], axis=1).to_numpy()
# 3 sexes, Male, Female, Infant

#normalize dataset to be positive values between 0 and 1
#decision to normalize to positive values only is because all attributes are positive measurements
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#normalize target variable(age)
#we shall be storing the min/max to rescale data later
min_y = min(y)
max_y = max(y)
y = (y - min_y) / (max_y - min_y)

#%% split data into training and test sets
from sklearn.model_selection import train_test_split

#using 80% of data for training, 20% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% random forest training
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=4, n_estimators=50)
rf.fit(X_train, y_train)

#%% evaluation on training set
y_pred = rf.predict(X_train)

#using mean squared error for evaluation
from sklearn.metrics import mean_squared_error
for i in range(len(y_train)):
    print(y_train[i], y_pred[i])
print(mean_squared_error(y_train, y_pred))
    
#%% evaluation on test set
y_pred = rf.predict(X_test)

for i in range(len(y_test)):
    print(y_test[i], y_pred[i])
print(mean_squared_error(y_test, y_pred))

#%% gradient boosting training

