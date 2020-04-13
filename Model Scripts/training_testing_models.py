"""
@author: Agni
"""

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

# Training and testing regression models
filepath = r'C:\Users\Agni\Documents\MIT 2017-18\6.884\neuro-soc\Data\neuro_dataset.xlsx'
df_original = pd.read_excel(filepath, sheetname=0)

df = df_original.copy() # for weekly examination
df['Name'] = 'Patient'
df['Stuff'] = 20
df['Quantity'] = 1
df['Visit Initiated'] = pd.to_datetime(df['Visit Initiated']) - pd.to_timedelta(7, unit='d')
df = df.groupby(['Name', pd.Grouper(key='Visit Initiated', freq='W-MON')])['Quantity'].sum().reset_index().sort_values('Visit Initiated')

df['date_delta'] = (df['Visit Initiated'] - df['Visit Initiated'].min())  / np.timedelta64(1,'D')

#df_times = df['date_delta'].copy() # used for this

df_times = df.filter(['date_delta','Stuff'], axis=1)
#df_times = df.drop(['Visit Initiated'], axis=1)
y = df['Quantity'].copy()

print("DF_TIMES:", df_times)

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df_times, y, test_size=0.2) # df or df_times?
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print(predictions[0:5])

print("Score:", model.score(X_test, y_test))

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

# Cross-validation (to be entered below)