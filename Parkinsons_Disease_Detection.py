# importing the dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Data Collection and analysis
# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons_data.csv')

# printing the first 5 rows of the dataframe

parkinsons_data.head()

# number of rows and columns in the dataframe
parkinsons_data.shape

# getting more information about the dataset
parkinsons_data.info()

# checking for missing values in each column
parkinsons_data.isnull().sum()

# getting some statistical measures about the data
parkinsons_data.describe()

# distribution of target Variable
parkinsons_data['status'].value_counts()

# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()

# Data Preprocessing
# separating the features and target
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Data Standardization
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)

# X_test = scaler.transform(X_test)
# print(X_train)

# Feature Selection with correlation
# X_train.corr()

# # sns.heatmap(df_corr, cmap='viridis', linecolor='k', linewidths=2, annot=True)

# plt.figure(figsize=(25,15))
# cor = X_train.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
# plt.show()

# def correlation(dataset, threshold):
#     col_corr = set()  # Set of all the names of correlated columns
#     corr_matrix = dataset.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
#                 colname = corr_matrix.columns[i]  # getting the name of column
#                 col_corr.add(colname)
#     return col_corr

# corr_features = correlation(X_train, 0.9)
# len(set(corr_features))

# corr_features

# X_train = X_train.drop(corr_features,axis=1)
# X_test = X_test.drop(corr_features,axis=1)

# X_train.columns

# X_train.count()

# Model Training
# Support Vector Machine Model
model = svm.SVC(kernel='linear')
# training the SVM model with training data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy Score
# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

# Building a Predictive System
input_data = (119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# # standardize the data
# std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")

# Saving the trained model
import pickle
filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
for column in X_train.columns:
  print(column)

import sklearn
print(sklearn.__version__)