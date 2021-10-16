import math
import sys
from threading import Thread
import numpy
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
_code_done = 0
_code_diverged = 1
_code_interrupted = 2
_code = 0
_loss = 1
_theta = 2
_iterations = 3

def main():
    trainData = readCSVFile('train.csv')  # Contains price labels
    YTrain = readCSVFile('train.csv')
    testData = readCSVFile('test.csv')
    cols1 = [1, 4, 17, 18, 19, 43, 44, 46, 62, 77]  # Features selected
    selectedColumns(testData, cols1)
    selectedColumns(trainData, cols1)
    selectedColumns(YTrain, [80])
    #fit(YTrain, cols1, testData, trainData)
    thread_1 = Thread(target=fit,name='Thread-1', args=(YTrain, cols1, testData, trainData))
    thread_2 = Thread(target=fit,name='Thread-2', args=(YTrain, cols1, testData, trainData))
    thread_1.start()
    thread_2.start()
    thread_2.join()
    thread_1.join()

    return 0


def fit(YTrain, cols1, testData, trainData, lr = 0.00000001, i=300000):
    # for row in trainData: print(row)
    trainData = toNumpyMat(trainData)
    testData = toNumpyMat(testData)
    yTrain = toNumpyMat(YTrain)
    # for row in testData: print(row)
    # use Sum of Squared diff SSD for cost func
    # Goal is to fit line/s over the train data and get the minimum error (Cost function value) using test data.
    # We try to automate this process by changing lr and selecting the best fit.
    # Theta mat start as random. Cost func J(theta1, theta2, ...) or J(theta)
    # for gradiant  [hypo - (actual y)]
    loss = []
    theta = np.random.rand(len(cols1) + 1)
    ones = np.ones(len(trainData))
    trainData = trainData.transpose()
    theta = theta
    trainData = np.vstack([ones, trainData])  # Now in shape and intercept ones
    counter = 0
    currLoss = 0
    return_code = _code_done
    while counter < i:
        try:
            Y = hypLinear(trainData.transpose(), theta)  # Hypothesis
            dJ = (trainData * (Y - YTrain).transpose()) / trainData.shape[1]  # Gradiant
            theta = theta - (lr * dJ.transpose())  # Update theta
            currLoss = ((Y - YTrain) ** 2).sum()
            if currLoss == math.inf:
                print('\nDiverged with lr=%f' % lr)
                return_code = _code_diverged
                break # Diverged
            counter += 1
            if counter % 100 == 0: print('i: %d, loss = %f, theta: %s' % (counter, currLoss, theta[0]))
        except KeyboardInterrupt:
            return_code = _code_interrupted
            break
    return return_code, currLoss, theta[0], counter

def hypLinear(XTrain, theta) -> numpy.ndarray:
    return theta * XTrain


def printFeatureIndcies(data):
    for i, val in enumerate(data[0]):
        print((i, val))


def readCSVFile(filename: str) -> list:
    import csv
    data = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for i, row in enumerate(csvReader):
            if i == 0:
                continue
            else:
                data.append(row)
    return data


def selectedColumns(data: list, columns: list):
    # Remove data not included in study
    # Need Cols 1, 4, 17, 18, 19, 43, 44, 46, 62, 77
    # This index base includes ID col.
    columns = set(columns)
    for rowIndex in range(len(data)):
        newRow = list()
        for featureIndex in range(len(data[rowIndex])):
            if featureIndex in columns:
                try:
                    newRow.append(int(data[rowIndex][featureIndex]))
                except ValueError:
                    newRow.append(0.0)
        # Replace new row
        data[rowIndex] = newRow


def toNumpyMat(data):
    return np.array(data)

    pass


if __name__ == '__main__':
    exit(main())
