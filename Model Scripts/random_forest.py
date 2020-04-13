"""
@author: Agni
"""

from sklearn.ensemble import RandomForestClassifier
import sklearn
import pandas as pd
import numpy as np
import librosa
import pickle

print('The scikit-learn version is {}.'.format(sklearn.__version__))

# LOADING DATA

"""
def getMFCC(path, filename):
    y, sr = librosa.load(path + filename)
    mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr)
    df = pd.DataFrame(mfcc_matrix).transpose()
    return df
"""

dataframes = []

"""
# For quick running and testing purposes (below code block is actual implementation)
path = "C:/Users/Agni/Desktop/verifai/audio/Truthful Audio Clips (Cleaned-Up)/"
template = 'trial_{}_{:03d}_clean.'
for i in range(1, 3): # truths m4a
    df = getMFCC(path, template.format('truth', i) + 'm4a')
    df[len(df.columns)] = 0 # truthful status
    dataframes.append(df)
    print('truth {} done'.format(i))
path = r"C:/Users/Agni/Desktop/verifai/audio/Deceptive Audio Clips (Cleaned-Up)/"
for i in range(1, 3): # lie mp3
    df = getMFCC(path, template.format('lie', i) + 'mp3')
    df[len(df.columns)] = 1 # deceptive status
    dataframes.append(df)
    print('lie {} done'.format(i))
"""

"""
path = "C:/Users/Agni/Desktop/verifai/audio/Truthful Audio Clips (Cleaned-Up)/"
template = 'trial_{}_{:03d}_clean.'
for i in range(1, 61): # truths m4a
    df = getMFCC(path, template.format('truth', i) + 'm4a')
    df[len(df.columns)] = 0 # truthful status
    dataframes.append(df)
    print('truth {} done'.format(i))
path = r"C:/Users/Agni/Desktop/verifai/audio/Deceptive Audio Clips (Cleaned-Up)/"
for i in range(1, 42): # lie mp3
    df = getMFCC(path, template.format('lie', i) + 'mp3')
    df[len(df.columns)] = 1 # deceptive status
    dataframes.append(df)
    print('lie {} done'.format(i))
for i in range(42, 62): # lie m4a
    df = getMFCC(path, template.format('lie', i) + 'm4a')
    df[len(df.columns)] = 1 # deceptive status
    dataframes.append(df)
    print('lie {} done'.format(i))
"""

df = pd.concat(dataframes)

column_names_list = []
for i in range(1, len(df.columns)):
    column_names_list.append("Feature " + str(i))
column_names_list.append("Status")
df.columns = column_names_list

# CREATING TRAINING AND TEST DATA
# Randomly sets 70% of rows in dataframe to be used for training (other 30% for testing) 
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .70
print(df.shape)
#print(df.columns)

# Creates two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# Shows the number of observations for the test and training dataframes
# print('Number of observations in the training data:', len(train))
# print('Number of observations in the test data:', len(test))

# DATA PREPROCESSING
# Creates a list of the feature columns' names
features = df.columns[:len(df.columns)-2] #up to and including last feature

# train['Status'] contains the actual species names. Convert 'Status' to digit (0 or 1)
y = pd.factorize(train['Status'])[0]

#TRAINING RANDOM FOREST CLASSIFIER
# Creates a random forest classifier
clf = RandomForestClassifier(n_jobs=1, random_state=0)

# Trains the classifier to take the training features and learn how they relate to the training y (the Status column)
clf.fit(train[features], y)
print("training done")

# APPLYING CLASSIFIER TO TEST DATA
# Applies the classifier trained to the test data
preds = clf.predict(test[features])
print("testing done")

# EVALUATING TEST CLASSIFIER
# print(preds[0:10])

# View the ACTUAL status (truthful or decptive) for the first few observations
# print(test['Status'].head())

# CONFUSION MATRIX (futher evalutaion of ML predictions)
# Create confusion matrix
confusion_matrix = pd.crosstab(test['Status'], preds, rownames=['Actual Status'], colnames=['Predicted Status'])
print(confusion_matrix)
pickle.dump(confusion_matrix, open("confusion_matrix.pkl", "wb"))
#confusion_matrix_pickled = pickle.load(open("confusion_matrix.pkl", "rb" ))
#print("PICKLED", confusion_matrix_pickled)

# VIEW FEATURE IMPORTANCE
# View a list of the features and their importance scores
importance_list = list(zip(train[features], clf.feature_importances_))
print(importance_list) 
pickle.dump(importance_list, open("importance_list.pkl", "wb"))