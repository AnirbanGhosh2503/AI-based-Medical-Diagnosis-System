# installing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set(rc={'figure.figsize': (20, 20)}, font_scale=1.4)

df = pd.read_csv('/content/hypothyroid.csv')
df

df.head()
df.describe().T
df.info()
df
df["binaryClass"].value_counts()
df["binaryClass"]=df["binaryClass"].map({"P":0,"N":1})
df["pregnant"].value_counts()
df=df.replace({"t":1,"f":0})
df
df['sex'].isnull().sum()
df["TBG"].value_counts()
del df["TBG"]
df=df.replace({"?":np.NAN})
df.isnull().sum()
df["sex"].value_counts()
df=df.replace({"F":1,"M":0})
df["referral source"].value_counts()
df = df.drop(["referral source"], axis=1)
df.info()
df["T3 measured"].value_counts()
df["TT4 measured"].value_counts()
df["FTI measured"].value_counts()
df["TBG measured"].value_counts()
df["binaryClass"].value_counts()
df.dtypes
cols = df.columns[df.dtypes.eq('object')]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df.dtypes
df.isnull().sum()
df['T4U measured'].mean()
df['T4U measured'].fillna(df['T4U measured'].mean(), inplace=True)
df['sex'].fillna(df['sex'].mean(), inplace=True)
df['age'].fillna(df['age'].mean(), inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df['TSH'] = imputer.fit_transform(df[['TSH']])
df['T3'] = imputer.fit_transform(df[['T3']])
df['TT4'] = imputer.fit_transform(df[['TT4']])
df['T4U'] = imputer.fit_transform(df[['T4U']])
df['FTI'] = imputer.fit_transform(df[['FTI']])
df.isnull().sum()
df
df.columns
df.to_csv('prepocessed_hyperthyroid.csv')
x = df.drop('binaryClass', axis=1)
y = df['binaryClass']
df.head()
x.head()
y.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x.shape, x_train.shape, x_test.shape)

# Feature Selection Dropping constant features
x_train = x_train.drop(['FTI', 'FTI measured', 'T4U measured', 'TT4 measured','query on thyroxine','on antithyroid medication','sick', 'pregnant','thyroid surgery','I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary','psych' , 'TSH measured', 'T4U', 'TBG measured'],axis=1)
x_test = x_test.drop(['FTI', 'FTI measured', 'T4U measured', 'TT4 measured','query on thyroxine','on antithyroid medication','sick', 'pregnant','thyroid surgery','I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary','psych' , 'TSH measured', 'T4U', 'TBG measured'],axis=1)

x_train.columns

# Model Training
model = LogisticRegression()
model.fit(x_train, y_train)

# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on Test data : ', test_data_accuracy)


input_data = (44,0,0,45,1,1.4,39)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a HyperThyroid Disease')
else:
  print('The Person has HyperThyroid Disease')


# Saving the trained model
import pickle
filename = 'Thyroid_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('Thyroid_model.sav', 'rb'))
for column in x_train.columns:
  print(column)
x_train.info() 