# Hoang-Nam Tran, z5629534
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, precision_score, mean_absolute_error
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import r_regression
import csv


# Data structure before preprocessing
# ['year', 'month', 'u10', 'v10', 'mx2t', 'mn2t', 'tcc', 't2', 'msl', 't', 'q', 'u', 'v', 'z', 'SPI', 'grid_ID']

# Data structure after preprocessing
# ['year', 'month', 'u10', 'v10', 'mx2t', 'mn2t', 'tcc', 't2', 'msl', 't', 'q', 'u', 'v', 'z', 'SPI', 'grid_ID','Drought', 'monthCos', 'monthSin']
variables = ['year', 'month', 'u10', 'v10', 'mx2t', 'mn2t', 'tcc', 't2', 'msl',
             't', 'q', 'u', 'v', 'z', 'SPI', 'grid_ID', 'Drought', 'monthCos', 'monthSin']

# forbidden indices: year, SPI, grid_ID, Drought -> 0, 14, 15, 16
forbiddenColumns = [0, 14, 15, 16]
seed = 42
keras.utils.set_random_seed(42)


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


def checkNonFloatNonNP(data):
    for x in data:
        for y in x:
            try:
                float(y)
            except:
                # print(y)
                data.remove(x)
                break


def filterInvalidMonthsNP(data):
    new_data = []
    validMonths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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


def detectOutliersByColumn(data, column):
    mean = np.mean(data[:, column])
    std_dev = np.std(data[:, column])
    z_scores = (data[:, column] - mean) / std_dev
    outliers = np.abs(z_scores) > 3
    outlier_indices = np.where(outliers)
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
    data = np.delete(data, outliersIndices, axis=0)
    return data


def plot_accuracy(result):
    # Extract accuracy and validation accuracy from the history object
    accuracy = result.history['accuracy']
    val_accuracy = result.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)

    # Plot the accuracy
    # plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    # plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.plot(result.history['accuracy'], 'b', label='Training accuracy')
    plt.plot(result.history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.ylim(0.8, 0.9)
    plt.legend()
    plt.show()


def plot_loss(result):
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(1, len(loss) + 1)

    #plt.plot(epochs, loss, 'b', label='Training loss')
    #plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.plot(result.history['loss'], 'b', label='Training loss')
    plt.plot(result.history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def scatterPlot(target, predicted):
    plt.scatter(target, predicted)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('SPI vs Predicted SPI')
    plt.show()


def plotSimpleConfusionMatrix(target, predicted):
    confusion_matrix = metrics.confusion_matrix(target, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=["No Drought", "Drought"])
    cm_display.plot()
    plt.show()


def printPredictorsSet(variables, indices):
    usedPredictors = []
    for x in indices:
        usedPredictors.append(variables[x])
    print("Used predictors: ", usedPredictors)


with open('Climate_SPI_Init.csv', newline='') as csvfile:
    initData = list(csv.reader(csvfile))

initData = drought(initData)
checkNonFloatNonNP(initData)
random.Random(seed).shuffle(initData)

initData = np.array(initData)
initData = initData.astype(float)
initData = removeOutliers(initData, forbiddenColumns)
initData = filterInvalidMonthsNP(initData)
initData = noIncludeInfinites(initData)
initData = normaliseMonth(initData)

# inputColumns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18]
# inputColumns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
# inputColumns = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
#inputColumns = [4, 6, 7, 8, 10, 11, 12, 13]


inputColumns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]

dataInput = initData[:, inputColumns]
dataTargetClassification = initData[:, 16]
dataTargetRegression = initData[:, 14]

scaler = MinMaxScaler()
scaler.fit(dataInput)
dataInputNormalized = scaler.transform(dataInput)
# dataInputNormalized = dataInput

inputTrain = dataInputNormalized[:int(0.7*len(initData))]
inputVal = dataInputNormalized[int(0.7*len(initData)):int(0.85*len(initData))]
inputTest = dataInputNormalized[int(0.85*len(initData)):]

targetTrainClass = dataTargetClassification[:int(0.7*len(initData))]
targetValClass = dataTargetClassification[int(
    0.7*len(initData)):int(0.85*len(initData))]
targetTestClass = dataTargetClassification[int(0.85*len(initData)):]

targetRegression = dataTargetRegression[:int(0.7*len(initData))]
targetValRegression = dataTargetRegression[int(
    0.7*len(initData)):int(0.85*len(initData))]
targetTestRegression = dataTargetRegression[int(0.85*len(initData)):]


initClassModel = Sequential()
# basic_model.add(Dense(units=16, activation='relu', input_shape=(13,)))
initClassModel.add(Dense(50, activation='relu', input_dim=14))
initClassModel.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(learning_rate=0.001)
initClassModel.compile(loss='binary_crossentropy',
                       optimizer=adam, metrics=["accuracy"])
resultClass = initClassModel.fit(inputTrain, targetTrainClass, epochs=150, batch_size=32, validation_data=(
    inputVal, targetValClass))  # used 150


adamReg = keras.optimizers.Adam(learning_rate=0.001)
regressionModel = Sequential()
# regressionModel.add(Dense(450, activation='relu', input_dim=13))
# regressionModel.add(Dense(90, activation= "relu"))
# regressionModel.add(Dense(45, activation= "relu"))
regressionModel.add(Dense(50, activation="relu", input_dim=14))
regressionModel.add(Dense(1))
regressionModel.compile(loss='mean_squared_error',
                        optimizer=adamReg, metrics=["mean_squared_error"])
resultRegression = regressionModel.fit(inputTrain, targetRegression, epochs=220,
                                       # used 220
                                       batch_size=32, validation_data=(inputVal, targetValRegression))

plot_accuracy(resultClass)

predictedClass = initClassModel.predict(inputTest)

# Adjust this line if your model outputs class labels directly
predictedClassBinary = (predictedClass >= 0.5).astype("int32")


plotSimpleConfusionMatrix(targetTestClass, predictedClassBinary)
print("Balanced Accuracy: ", balanced_accuracy_score(targetTestClass, predictedClassBinary))
print("Precision: ", precision_score(targetTestClass, predictedClassBinary, pos_label=1))


plot_loss(resultRegression)

predictedRegression = regressionModel.predict(inputTest)
scatterPlot(targetTestRegression, predictedRegression)
print("Mean Absolute Error: ", mean_absolute_error(
    targetTestRegression, predictedRegression))
print("Pearson Correlation Coefficient: ", r_regression(
    predictedRegression, targetTestRegression))


initClassModel.save("classification4.keras")
regressionModel.save("regression.keras")
classificationModel = keras.models.load_model("classification4.keras")
regressionModelLoaded = keras.models.load_model("regression.keras")


# might need to change file name during discussion
with open('Fake_Climate_SPI6.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

data = drought(data)
checkNonFloatNonNP(data)
random.Random(seed).shuffle(data)

data = np.array(data)
data = data.astype(float)
data = removeOutliers(data, forbiddenColumns)
data = filterInvalidMonthsNP(data)
data = noIncludeInfinites(data)
data = normaliseMonth(data)


dataInput = data[:, inputColumns]
dataTargetClassification = data[:, 16]
dataTargetRegression = data[:, 14]

scaler = MinMaxScaler()
scaler.fit(dataInput)
dataInputNormalized = scaler.transform(dataInput)

nClassificationPredicted = classificationModel.predict(dataInputNormalized)
nClassificationPredBin = (nClassificationPredicted >= 0.5).astype("int32")


plotSimpleConfusionMatrix(dataTargetClassification, nClassificationPredBin)
print("Balanced Accuracy: ", balanced_accuracy_score(
    dataTargetClassification, nClassificationPredBin))
print("Precision: ", precision_score(
    dataTargetClassification, nClassificationPredBin, pos_label=1))
print("Number of samples: ", len(data))
printPredictorsSet(variables, inputColumns)


nRegressionPredicted = regressionModelLoaded.predict(dataInputNormalized)

scatterPlot(dataTargetRegression, nRegressionPredicted)
print("Mean Absolute Error: ", mean_absolute_error(
    dataTargetRegression, nRegressionPredicted))
print("Pearson Correlation Coefficient: ", r_regression(
    nRegressionPredicted, dataTargetRegression))
print("Number of samples: ", len(data))
printPredictorsSet(variables, inputColumns)
