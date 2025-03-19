#importing the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

#Data Collection and Processing
# loading the csv data to a Pandas DataFrame
data = pd.read_csv('/content/survey lung cancer.csv')

# print first 5 rows of the dataset
data.head()

# print last 5 rows of the dataset
data.tail()

# number of rows and columns in the dataset
data.shape

# getting some info about the data
data.info()

data.describe()

data.isnull().sum()

# Label Encoder
label_encoder = preprocessing.LabelEncoder()

data['GENDER']= label_encoder.fit_transform(data['GENDER'])
  
data['GENDER'].unique()

data['LUNG_CANCER']= label_encoder.fit_transform(data['LUNG_CANCER'])
  
data['LUNG_CANCER'].unique()

# Here Male : Female = 1:0 & Cancer_positive : Cancer_negative = 1:0 
data.head(10)

data.tail()

# checking the distribution of Target Variable
data['LUNG_CANCER'].value_counts()

data.to_csv('prepocessed_lungs_data.csv')

# Splitting the features and target
X = data.drop(columns='LUNG_CANCER', axis=1)
Y = data['LUNG_CANCER']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training
# Linear Regression
model = LogisticRegression()
model.fit(X_train.values, Y_train.values)

# Model Evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)


input_data = (0,61,1,1,1,1,2,2,1,1,1,1,2,1,1)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Lung Cancer')
else:
  print('The Person has Lung Cancer')

# Saving the trained model
import pickle
filename = 'lungs_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))
for column in X_train.columns:
  print(column)