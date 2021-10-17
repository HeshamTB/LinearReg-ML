import math
import sys
import numpy as np
import csv

#np.set_printoptions(threshold=sys.maxsize)


def main():
    trainX_full = readCSVFile('train.csv')
    trainX = trainX_full
    cols1 = [1, 4, 17, 18, 19, 43, 44, 46, 62, 77]  # Features selected
    lr = 0.0000000001
    selectedColumns(trainX, cols1)
    trainX = np.matrix(trainX)
    trainX = trainX.transpose()
    ones = np.ones(trainX.shape[1])
    trainX = np.vstack([ones, trainX])  # Now in shape and intercept ones
    #print('trainX ', trainX.shape)
    trainY = readCSVFile('train.csv')
    selectedColumns(trainY, [80])
    trainY = np.matrix(trainY)
    trainY = trainY.transpose()
    #print('trainY ', trainY.shape)
    theta = np.random.random(trainX.shape[0])
    theta = np.matrix(theta)
    #print('theta ', theta.shape)
    fit(lr, theta, trainX, trainY)


def fit(lr, theta, trainX, trainY, iterations=500000):
    count = 0
    while count < iterations:
        Y = hypo(trainX, theta)
        # print(Y)
        # print('Y ', Y.shape)
        # print(trainX.shape[1])
        dJ = (trainX * (Y - trainY).transpose()) / trainX.shape[1]
        # print('dJ ', dJ.shape)
        theta = theta - lr * dJ.transpose()
        loss = ((Y - trainY).sum() ** 2) / trainX.shape[1]
        count += 1
        if loss == math.nan or loss == math.inf: break  # Diverged
        if count % 100 == 0: print('i=%d, loss=%d' % (count, loss))


def hypo(x, theta):
    return theta*x


def readCSVFile(filename: str) -> list:
    import csv
    data = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for i, row in enumerate(csvReader):
            if i == 0:
                continue ## pass first row (feat names)
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

def printFeatureIndcies(data):
    for i, val in enumerate(data[0]):
        print((i, val))

if __name__ == '__main__':
    exit(main())