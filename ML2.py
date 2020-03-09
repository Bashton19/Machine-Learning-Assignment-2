import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Section 1: Import, Summary and Preprocessing

# Reads data in
data = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv')
print()
'''
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
    categorical[var] = 1.*data[var].nunique()/data[var].count() < 0.5 # Calculates ratio of unique values
print("Are there any categorical values?")
print("==================================")
print(pd.DataFrame.from_dict(categorical, orient='index', columns=[' '])) # Converts dictionary into a dataframe to be printed
print()

# Create boxplot of vibration sensor 1, for both status classes
data.boxplot(column = 'Vibration_sensor_1', by = 'Status')
plt.title(' ') # Title provided by Pandas
plt.ylabel('Vibration Sensor 1') # y axis Label

# Create density plot of vibration sensor 2, for both status classes
plt.figure() # New figure
data.groupby('Status').Vibration_sensor_2.plot.kde(legend='True') 
plt.title('Density plot grouped by Status') # Title and x axis label
plt.xlabel('Vibration Sensor 2')
'''
# Section 3: Designing Algorithms

shuffle_data = data.sample(frac=1) # Returns full random sample of data
train_data = shuffle_data[0:(int(round(len(shuffle_data)*0.9)))] # Multiplies length of data by 90%, and rounds it
test_data = shuffle_data[(int(round(len(shuffle_data)*0.9))):(len(shuffle_data))] # Gets other 10% of data

x_train = train_data.loc[:,'Power_range_sensor_1':'Vibration_sensor_4'].values # Input variables for training assigned to x
y_train = train_data.loc[:,'Status'].values # Classes for training assigned to y

x_test = test_data.loc[:,'Power_range_sensor_1':'Vibration_sensor_4'].values # Input variables for testing assigned to x
y_test = test_data.loc[:,'Status'].values # Predicted labels to compare with true labels assigned to y

'''
Artificial Neural Network
solver determines how weights are optimised
Sigmoid activation function for hidden layers: f(x) = 1 / (1 + exp(-x))
500 max iterations, though it will stop if convergence occurs before
2 hidden layers with 500 neurons 
'''
'''
NeuralNet = MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(500, 500))

NeuralNet.fit(x_train, y_train) # Trains neural network

y_pred=NeuralNet.predict(x_test) # Predicts and classifies x_test

accuracy_nn = accuracy_score(y_test, y_pred) # Calculates rate of correctly classified cases

confusion_nn = confusion_matrix(y_test, y_pred) # Creates confusion matrix

print()
print("Classifications of Neural Network:")
print(y_pred)
print()
print("Accuracy of Neural Network:")
print(accuracy_nn)
print()
print("Confusion Matrix:")
print(confusion_nn)

# Bonus Task: Epochs vs Accuracy
epochs = [10, 20, 30, 40, 50, 100, 250, 500, 750, 1000, 2000, 5000, 10000] # List of No. of Epochs
NN_accuracies = [] # Create list to store accuracy scores
for i in epochs: # For each value in list
    NeuralNet = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=i) # Set's max epochs to i
    setattr(NeuralNet, "out_activation_", "logistic") # Sets output function to Sigmoid
    NeuralNet.fit(x_train, y_train) # Trains network
    NN_y_pred=NeuralNet.predict(x_test) # Predicts values
    NN_accuracy=accuracy_score(y_test, NN_y_pred) # Calculates accuracy score
    NN_accuracies.append(NN_accuracy) # Appends every score to list 
    
plt.figure() # Creates new figure    
plt.plot(epochs, NN_accuracies) # Plots list of epoch values vs corresponding accuracy scores
plt.ylim(0.6, 1.0) # Sets y axis limit for clearer look
plt.ylabel('Accuracy Score') # Y axis label
plt.xlabel('Epochs') # X label axis
plt.title('Neural Network: Epochs vs Accuracy Score') # Title of plot


# Random forest classifier of 1000 trees, with minimum number of samples required to be at leaf node of 50
RandomForest = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 5)

RandomForest.fit(x_train, y_train) # Trains model using training data

y_pred2 = RandomForest.predict(x_test) # Predicts and classifies x_test

accuracy_rf = accuracy_score(y_test, y_pred2) # Calculates rate of correctly classified cases
confusion_rf = confusion_matrix(y_test, y_pred2) # Creates confusion matrix

print()
print("Classifications of Random Forest:")
print(y_pred2)
print()
print("Accuracy of Random Forest:")
print(accuracy_rf)
print()
print("Confusion Matrix:")
print(confusion_rf)

# Bonus Task: Trees vs Accuracy
trees = [10, 20, 30, 40, 50, 100, 250, 500, 750, 1000, 2000, 5000, 7500] # List of Tree values to be passed into RandomForest
RF_accuracies=[] # List to store accuracies for first model
RF_accuracies2 = [] # List to store accuracies for second model
for i in trees: # For each number of trees in list
    RandomForest = RandomForestClassifier(n_estimators=i, min_samples_leaf = 5) # Model with 5 minimum samples required for leaf node
    RandomForest2 = RandomForestClassifier(n_estimators=i, min_samples_leaf = 50) # Model with 50 minimum samples required for leaf node
    RandomForest.fit(x_train, y_train) # Trains the first model
    RandomForest2.fit(x_train, y_train) # Trains the second model   
    RF_y_pred = RandomForest.predict(x_test) # Classifies test data for first model
    RF_y_pred2 = RandomForest2.predict(x_test) # Classifies test data for second model
    RF_accuracy = accuracy_score(y_test, RF_y_pred) # Calculates accuracy for first model
    RF_accuracy2 = accuracy_score(y_test, RF_y_pred2) # Calculates accuracy for second model
    RF_accuracies.append(RF_accuracy) # Appends accuracy scores to list
    RF_accuracies2.append(RF_accuracy2) # Appends accuracy scores to list
    
plt.figure() # Creates a new figure
plt.plot(trees, RF_accuracies, 'b') # Plots number of trees vs. accuracy scores for first model
plt.plot(trees, RF_accuracies2, 'r') # Plots no. of trees vs. accuracy scores for second model
plt.legend(('Minimum Sample for Leaf Node = 5', 'Minimum Sample for Leaf Node = 50')) # Legend
plt.ylabel('Accuracy Score') # Labels y-axis
plt.xlabel('Trees') # Labels x-axis
plt.ylim(0.6, 1.0) # Limits y-axis for clearer look
plt.title('Random Forest: Trees vs Accuracy Score') # Title
'''

# Section 4: Model Selection

data = data.sample(frac=1) # Returns full random sample of data

X = data.loc[:,'Power_range_sensor_1':'Vibration_sensor_4'].values # Splits features from data
y = data.loc[:,'Status'].values # Splits classes from data

neurons=[50, 500, 1000] # List of number of neurons to be tested
for i in neurons: # For each value in list
    MLP = MLPClassifier(activation='logistic', max_iter=500, hidden_layer_sizes=(i, i)) # Initialise MLP with neurons i
    setattr(MLP, "out_activation_", "logistic") # Output function to sigmoid
    MLP_CV = cross_val_score(MLP, X, y, cv = 10) # Cross validate MLP with 10 k folds
    print("Artificial neural network with ", i, " neurons in hidden layers") # Print number of neurons used
    print()
    print("Cross validation score:")
    print(MLP_CV) # Print accuracy of each fold
    print()
    print("Mean accuracy: ", np.mean(MLP_CV)) # Print mean of accuracies
    print()

trees=[20, 500, 10000] # List of number of trees to be tested
for i in trees: # For each value in list
    RF = RandomForestClassifier(n_estimators=i) # Initialise random forest classifier with trees i
    RF_CV = cross_val_score(RF, X, y, cv = 10) # Cross validate classifier with 10 k folds
    print("Random forest with", i, "trees") # Print number of trees
    print()
    print("Cross validation score:")
    print(RF_CV) # Print accuracy of each fold
    print()
    print("Mean accuracy: ", np.mean(RF_CV)) # Print mean of accuracies
    print()


        