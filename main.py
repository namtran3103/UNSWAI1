import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score

from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import csv


#numpy stretching and slicing    
    
#print size of data[0]
#Data [['year', 'month', 'u10', 'v10', 'mx2t', 'mn2t', 'tcc', 't2', 'msl', 't', 'q', 'u', 'v', 'z', 'SPI', 'grid_ID'],...]
#print((data[0]))

#a) SPI to Drought, SPI <=-1 is drought, SPI >-1 is no drought
#modify data list by adding drought variable, später vll entfernen erste zeile data[0]
#data[0].append('Drought')
#get rid of first row

#Data [['year', 'month', 'u10', 'v10', 'mx2t', 'mn2t', 'tcc', 't2', 'msl', 't', 'q', 'u', 'v', 'z', 'SPI', 'grid_ID','Drought'],...]
#forbidden indices: SPI, Drought, year, grid_ID -> 14, 15, 0, 16

forbiddenColumns = [0, 14, 15, 16]

def drought(data):
    for x in data:
        try:
            if float(x[14]) <= -1:
                x.append(1)
            else:
                x.append(0)
        except:
            x.append('Drought')
    return data

'''
def check_np_empty_nonNum_values(data):
    for x in data:
        for y in x:
            try:
                float(y)
            except:
                index = np.where(data == x)
                #print the row
                print(index)
                data = np.delete(data, index, axis=0)
                break
    return data
'''
def checkNonFloatNonNP(data):
    for x in data:
        for y in x:
            try:
                float(y)
            except:
                print(y)
                data.remove(x)
                break

def filterInvalidMonthsNP(data):
    new_data = []
    validMonths = [1,2,3,4,5,6,7,8,9,10,11,12]
    for x in data:
        if x[1] in validMonths:
            new_data.append(x)
    return np.array(new_data)

def lineCount(data):
    count = 0
    for x in data:
        count += 1
    return count

def normaliseMonth(dataSet):
    new_dataSet = []
    for x in dataSet:
        month = x[1]
        month_normalised = 2 * np.pi * (month - 1) / 12
        x = np.append(x, [np.cos(month_normalised), np.sin(month_normalised)])
        #new_dataSet.append(np.delete(x, 1))
        new_dataSet.append(x)
    return np.array(new_dataSet)

def noIncludeInfinites(data):
    new_data = []
    detected = False
    for x in data:
        for y in x:
            if y == float('inf') or y == float('-inf'):
                detected = True
                break
        if detected == False:
            new_data.append(x)
    return np.array(new_data)

def detect_outliers(data):    
    # Calculate the mean and standard deviation for each column
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    
    
    # Calculate the Z-score for each data point
    z_scores = (data - mean) / std_dev
    
    # Identify outliers (Z-score > 3 or Z-score < -3)
    outliers = np.abs(z_scores) > 3
    #print(outliers)
    
    # Get the indices of the outliers
    outlier_indices = np.where(outliers)
    
    #count how many outliers
    count = 0
    for x in outliers:
        for y in x:
            if y == True:
                count += 1
                break
    print(count)
    
    county = 0
    for x in outlier_indices[1]:
        if x in forbiddenColumns:
            county += 1
    print(county)
    print(count-county)
    return outlier_indices

def detectOutliersByColumn(data, column):
    mean = np.mean(data[:, column])
    std_dev = np.std(data[:, column])
    z_scores = (data[:, column] - mean) / std_dev
    outliers = np.abs(z_scores) > 3
    outlier_indices = np.where(outliers)
    #count and print outlierindices
    #print(len(outlier_indices[0]))
    return outlier_indices

def removeOutliers(data, excludedColumns):
    outliersIndices = set()
    for i in range(0, len(data[0])):
        if i not in excludedColumns:
            outlierIndicesColumn = detectOutliersByColumn(data, i)
            for x in outlierIndicesColumn[0]:
                outliersIndices.add(x)

    outliersIndices = list(outliersIndices)
    outliersIndices.sort()
    #print(outliersIndices)
    data = np.delete(data, outliersIndices, axis=0)
    return data

def plot_accuracy(result):
    # Extract accuracy and validation accuracy from the history object
    accuracy = result.history['accuracy']
    val_accuracy = result.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    
    # Plot the accuracy
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'ro', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.ylim(0, 1) 
    plt.legend()
    plt.show()
    
def plotSimpleConfusionMatrix(target, predicted):
    confusion_matrix = metrics.confusion_matrix(target, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["No Drought", "Drought"])
    cm_display.plot()
    plt.show()     
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_performance(y_true, y_pred):
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)  # Assuming '1' is the positive class

    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print("Balanced Accuracy: ", balanced_accuracy)
    print("Precision: ", precision)

with open('Climate_SPI.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

data = drought(data)
print(data[0])
print(lineCount(data))
checkNonFloatNonNP(data)
print(lineCount(data))
#np array of data
data = np.array(data)
#preprossing 1, get rid of rows that contain empty values or non numeric values

#data = check_np_empty_nonNum_values(data)
print(lineCount(data))
data = data.astype(float)
#(b) Split your data into training, validation and test sets.
#split data into 80% training, 10% validation and 10% test, shuffle data

#np.random.shuffle(data)

#outliers = detect_outliers(data)
#print(outliers)



data = removeOutliers(data, forbiddenColumns)

print(lineCount(data))
print(data[999])
data = filterInvalidMonthsNP(data)
data = noIncludeInfinites(data)
data = normaliseMonth(data)
print(lineCount(data))
print(data[999])

#np.random.shuffle(data)




data = filterInvalidMonthsNP(data)
print(lineCount(data))



train_data = data[:int(0.7*len(data))]
val_data = data[int(0.7*len(data)):int(0.85*len(data))]
test_data = data[int(0.85*len(data)):]




#select columns 2-13 and 17 for input

inputColumns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18]
#inputColumns = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
input = train_data[:, inputColumns] 
target = train_data[:,16]
inputVal = val_data[:, inputColumns]
targetVal = val_data[:,16]
inputTest = test_data[:, inputColumns]
targetTest = test_data[:,16]

scaler1 = MinMaxScaler()
scaler1.fit(input)
t_input = scaler1.transform(input)
scaler2 = MinMaxScaler()
scaler2.fit(inputVal)
t_inputVal = scaler2.transform(inputVal)
scaler3 = MinMaxScaler()
scaler3.fit(inputTest)
t_inputTest = scaler3.transform(inputTest)

basic_model = Sequential()
basic_model.add(Dense(units=16, activation='relu', input_shape=(13,)))
basic_model.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(learning_rate=0.001)
basic_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

result = basic_model.fit(t_input, target, epochs=150, batch_size=30, validation_data=(t_inputVal, targetVal))



plot_accuracy(result)

predicted = basic_model.predict(t_inputTest)
#predicted = tf.squeeze(predicted)
#predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
#actual = np.array(targetTest)
#conf_mat = confusion_matrix(actual, predicted)
#displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
#displ.plot()



#print(y_pred)
#print(y_test)

y_pred = (predicted >= 0.5).astype("int32")  # Adjust this line if your model outputs class labels directly

# Evaluate the performance
#evaluate_performance(y_test, y_pred)

# Compute and plot the confusion matrix
class_names = ['No Drought', 'Drought']  # Adjust class names as needed
#plot_confusion_matrix(targetTest, y_pred, class_names)
plotSimpleConfusionMatrix(targetTest, y_pred)

#evaluate_performance(y_test, y_pred)


'''
# define the keras model
model = Sequential()
model.add(Dense(500, input_shape=(13,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
result = model.fit(input, target, epochs=250, batch_size=100, validation_data=(inputVal, targetVal))
# evaluate the keras model
_, accuracy = model.evaluate(input, target)
print('Accuracy: %.2f' % (accuracy*100))

_, accuracyVal = model.evaluate(inputVal, targetVal)
print('Accuracy: %.2f' % (accuracyVal*100))

#Plot of the accuracy (y-axis) versus the number of epochs (x-axis) for both the training and validation sets





test_input = test_data[:, inputColumns]

#Performance metrics “Balanced Accuracy” and “Precision” calculated on the test set.

# Predict the class on the test set
y_pred = model.predict(test_input)
y_test = test_data[:, 16]

#print(y_pred)
#print(y_test)
y_pred = (y_pred > 0.5).astype("int32")  # Adjust this line if your model outputs class labels directly

# Evaluate the performance
#evaluate_performance(y_test, y_pred)

# Compute and plot the confusion matrix
class_names = ['No Drought', 'Drought']  # Adjust class names as needed
plot_confusion_matrix(y_test, y_pred, class_names)


evaluate_performance(y_test, y_pred)

plot_accuracy(result)


#print(lineCount(train_data)+lineCount(val_data)+lineCount(test_data))   

'''

#print row of each
#print(train_data[0])

#print data type of each single val
#print(type(train_data[2][3]))
#(c) Pre-processing: Apply any necessary transformation to the trainingset, then apply the same transformation to the validation and test sets.
#convert data to float

#print(train_data[0])
#check for any empty values and non numeric values in a row and delete the row 

'''
def check_empty_values(data):
    
    
check_empty_values(train_data)
check_empty_values(val_data)    
check_empty_values(test_data)
'''
#– Normalise the month to the range [0, 2π] using: month normalised = 2π× (month - 1)/12., 
# replace ’month’ with two new predictors: ‘cos(month normalised)’ and ‘sin(month normalised)’.
#Data [['year', 'month', 'u10', 'v10', 'mx2t', 'mn2t', 'tcc', 't2', 'msl', 't', 'q', 'u', 'v', 'z', 'SPI', 'grid_ID','Drought', 'cosMonthNorm', 'sinMonthNorm],...]


#print(train_data[0])

#train_data = normaliseMonth(train_data)
#val_data = normaliseMonth(val_data)
#test_data = normaliseMonth(test_data)
#print(type(train_data[0][0]))
#print(train_data[0])




