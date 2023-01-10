
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data3.csv')
X=data.drop(columns=['BPA','id'], axis=1)
Y=data['BPA']

#train and test Input data
X_train , X_test , Y_train , Y_test =  train_test_split(X,Y,test_size=0.3)

#perform Scaling
from sklearn.preprocessing import RobustScaler
rbX = RobustScaler()
X_train = rbX.fit_transform(X_train)
X_test = rbX.transform(X_test)


#Fit the model Classifier
classifier = RandomForestClassifier()
classifier = classifier.fit(X_train, Y_train)

#input data 
input_data = (11.28,0.9,34,23,1,1,0,45961,48071,397,2,1,1)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = rbX.fit_transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)




