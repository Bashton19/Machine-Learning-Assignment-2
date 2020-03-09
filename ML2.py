import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Section 1: Import, Summary and Preprocessing

# Reads data in

data = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv')

# Summary Statistics
'''
print("")
print("Mean")
print("")
print(np.mean(data))
print("")
print("Standard Deviation")
print("")
print(np.std(data))
print("")
print("Minimum")
print("")
print(np.min(data))
print("")
print("Maximum")
print("")
print(np.max(data))
print("")
print("Shape and size of dataset")
print (data.shape)
print (data.size)
print("")

# Checks for missing values

data.isnull()

# Checks for categorical variables

categorical = {}
for var in data.columns:
    categorical[var] = 1.*data[var].nunique()/data[var].count() < 0.05
    
print(categorical)

# Creates boxplot

data.boxplot(column = 'Vibration_sensor_1', by = 'Status')
plt.title('')
plt.ylabel('Vibration Sensor 1')

# Creates density plot

plt.figure()
data.groupby('Status').Vibration_sensor_2.plot.kde(legend='True')
plt.title('Density plot grouped by Status')
plt.xlabel('Vibration Sensor 2')
'''
# Section 3: Designing Algorithms

# ANN with sklearn

shuffle_data = data.sample(frac=1)
train_data = shuffle_data[0:(int(round(len(shuffle_data)*0.9)))]
test_data = shuffle_data[(int(round(len(shuffle_data)*0.9))):(len(shuffle_data))]

x_train = train_data.iloc[:,1:12].values
y_train = train_data.loc[:,'Status'].values

x_test = test_data.iloc[:,1:12].values
y_test = test_data.loc[:,'Status'].values

NeuralNet = MLPClassifier(solver = 'sgd', alpha = 0.0001, learning_rate='adaptive',
                    activation='logistic', max_iter=10, hidden_layer_sizes=(500, 500), random_state=1)

NeuralNet.fit(x_train, y_train) # Trains neural network

y_pred=NeuralNet.predict(x_test) # Classifies x_test

accuracy_nn = accuracy_score(y_test, y_pred) # Calculates accuracy of predictions

plt.figure()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

print("")
print("Classifications of Neural Network: ", y_pred)
print("")
print("Accuracy of Neural Network: ", accuracy_nn)

# Random forest classifier

RandomForest = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 50)

RandomForest.fit(x_train, y_train)

y_pred2 = RandomForest.predict(x_test)

accuracy_rf = accuracy_score(y_test, y_pred2) # Calculates accuracy of predictions

print("")
print("Classifications of Random Forest: ", y_pred2)
print("")
print("Accuracy of Random Forest: ", accuracy_rf) 
    
# Section 4: Model Selection

        
        
        

        
        