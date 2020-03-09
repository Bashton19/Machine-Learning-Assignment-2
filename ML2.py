import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# Section 1: Import, Summary and Preprocessing

# Reads data in
data = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv')
print()

# Summary Statistics
print("Mean")
print("=====")
print(np.mean(data)) # Calculates mean average of each feature
print()
print("Variance")
print("=========")
print(np.var(data)) # Calculates variance of each feature
print()
print("Standard Deviation") 
print("===================")
print(np.std(data)) # Calculates standard deviation of each feature
print()
print("Minimum")
print("========")
print(np.min(data)) # Finds minimum of feature
print()
print("Maximum")
print("========")
print(np.max(data)) # Finds maximum of each feature
print()

print("Shape and size of dataset")
print("==========================")
print (data.shape) # Dimensions of dataset
print (data.size) # Total size of dataset
print()

print("Are there any missing values in any columns?")
print("=============================================")
print(data.isnull().any()) # Checks for missing values
print()

# Checks for categorical variables
categorical = {} # Creates dictionary
for var in data.columns: # For each column
    categorical[var] = 1.*data[var].nunique()/data[var].count() < 0.05 # Calculates ratio of unique values
print("Are there any categorical values?")
print("==================================")
print(pd.DataFrame.from_dict(categorical, orient='index', columns=[' '])) # Converts dictionary into a dataframe to be printed
print()

# Create boxplot of vibration sensor 1, for both status classes
data.boxplot(column = 'Vibration_sensor_1', by = 'Status')
plt.title('')
plt.ylabel('Vibration Sensor 1') # y axis Label

# Create density plot of vibration sensor 2, for both status classes
plt.figure() # New figure
data.groupby('Status').Vibration_sensor_2.plot.kde(legend='True') 
plt.title('Density plot grouped by Status') # Title and x axis label
plt.xlabel('Vibration Sensor 2')

# Section 3: Designing Algorithms

# ANN with sklearn

shuffle_data = data.sample(frac=1) # Returns full random sample of data
train_data = shuffle_data[0:(int(round(len(shuffle_data)*0.9)))] # Multiplies length of data by 90%, and rounds it
test_data = shuffle_data[(int(round(len(shuffle_data)*0.9))):(len(shuffle_data))] # Gets other 10% of data

x_train = train_data.iloc[:,1:12].values # Input variables for training assigned to x
y_train = train_data.loc[:,'Status'].values # Classes for training assigned to y

x_test = test_data.loc[:,1:12].values # Input variables for testing assigned to x
y_test = test_data.loc[:,'Status'].values # Predicted labels to compare with true labels assigned to y

'''
Artificial Neural Network
solver determines how weights are optimised
Sigmoid activation function for hidden layers: f(x) = 1 / (1 + exp(-x))
500 max iterations, though it will stop if convergence occurs before
2 hidden layers with 500 neurons
'''

NeuralNet = MLPClassifier(solver = 'lbfgs', # Creates neural network classifier
                    activation='logistic', max_iter=500, hidden_layer_sizes=(500, 500), random_state=0)

NeuralNet.fit(x_train, y_train) # Trains neural network

y_pred=NeuralNet.predict(x_test) # Predicts and classifies x_test

accuracy_nn = accuracy_score(y_test, y_pred) # Calculates rate of correctly classified cases

'''
plt.figure()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
'''

print()
print("Classifications of Neural Network: ", y_pred) # Prints predicted labels
print()
print("Accuracy of Neural Network: ", accuracy_nn) # Prints accuracy

# Random forest classifier of 1000 trees, with minimum number of samples required to be at leaf node of 50
RandomForest = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 5)

RandomForest.fit(x_train, y_train) # Trains model using training data

y_pred2 = RandomForest.predict(x_test) # Predicts and classifies x_test

accuracy_rf = accuracy_score(y_test, y_pred2) # Calculates rate of correctly of correctly classified cases

print()
print("Classifications of Random Forest: ", y_pred2) # Prints predicted labels
print()
print("Accuracy of Random Forest: ", accuracy_rf) # Prints accuracy
    
# Section 4: Model Selection

kf = KFold(n_splits = 10, shuffle=True) # 
kf.get_n_splits(data)
        
        

        
        