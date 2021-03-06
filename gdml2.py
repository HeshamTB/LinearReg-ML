import math
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
order = 1

def main():
    cols1 = [1, 4, 17, 18, 19, 43, 44, 46, 62, 77]  # Features selected
    if sys.argv[1] == 'learn':
        trainX_full = readCSVFile('train.csv')
        trainX = trainX_full
        lr = float(sys.argv[2])
        selectedColumns(trainX, cols1)
        trainX = raiseOrder(trainX, order)
        trainX = featurescaling(trainX)
        trainX = np.matrix(trainX)
        trainX = trainX.transpose()
        ones = np.ones(trainX.shape[1])
        trainX = np.vstack([ones, trainX])  # Now in shape and intercept ones
        # print('trainX ', trainX.shape)
        trainY = readCSVFile('train.csv')
        selectedColumns(trainY, [80])
        #trainY = featurescaling(trainY)
        trainY = np.matrix(trainY)
        trainY = trainY.transpose()
        # print('trainY ', trainY.shape)
        theta = np.random.random(trainX.shape[0])
        theta = np.matrix(theta)
        # print('theta ', theta.shape)
        loss, theta = fit(lr, theta, trainX, trainY, True, int(sys.argv[3]))
        # print(loss, theta)
        # Test
        testX = readCSVFile('test.csv')
        # printFeatureIndcies(testX)
        selectedColumns(testX, cols1)
        testX = raiseOrder(testX, order)
        testX = featurescaling(testX)
        testX = np.matrix(testX)
        testX = testX.transpose()
        ones = np.ones(testX.shape[1])
        testX = np.vstack([ones, testX])  # Now in shape and intercept ones
        testY = readCSVFile('test.csv')
        selectedColumns(testY, [80])
        testY = np.matrix(testY)
        testY = testY.transpose()
        loss_test, Y_infer = test(testX, testY, theta)
        print(Y_infer)
        print('\n\n', loss_test, theta)

    elif sys.argv[1] == 'test':
        theta = sys.argv[2].split()
        print(theta)

def test(testX, testY, theta):
    Y_infer = hypo(testX, theta)
    loss_test = ((Y_infer - testY).sum() ** 2) / testX.shape[1]
    return loss_test, Y_infer


def fit(lr, theta, trainX, trainY, verbose, iterations=500000):
    count = 0
    min = math.inf
    loss = 0
    while count < iterations:
        Y = hypo(trainX, theta)
        dJ = (trainX * (Y - trainY).transpose()) / trainX.shape[1]
        theta = theta - lr * dJ.transpose()
        loss = ((Y - trainY).sum() ** 2) / trainX.shape[1]
        last = loss
        try:
            Y = hypo(trainX, theta)
            dJ = (trainX * (Y - trainY).transpose()) / trainX.shape[1]
            theta = theta - lr * dJ.transpose()
            loss = ((Y - trainY).sum() ** 2) / trainX.shape[1]
            count += 1
            if loss == math.nan or loss == math.inf: break  # Diverged
            if count % 1 == 0 and verbose: print('i=%d, loss=%f, theta=%s' % (count, loss, theta))
        except KeyboardInterrupt:
            break
    # return data
    return loss, theta


def raiseOrder(data: list[list], order: int):
    data_new = list()
    for i, row in enumerate(data):
        new_row = list()
        for j, elemnt in enumerate(row):
            for k in range(order):
                new_row.append(elemnt ** (k+1))
        data_new.append(new_row)
    return data_new

def hypo(x : np.matrix, theta : np.matrix) -> np.matrix:
    return theta * x


def featurescaling(array):
    #  feature scaling
    biggest_value = 0
    counter_list = 0
    counter_element = 0
    for mnm in range(len(array[0])):
        for me in array:
            if (int(array[counter_list][counter_element])) > biggest_value:
                biggest_value = (int(array[counter_list][counter_element]))
            counter_list = counter_list + 1
        counter_list = 0

        for mee in array:
            array[counter_list][counter_element] = ((int(array[counter_list][counter_element])) / biggest_value)
            counter_list = counter_list + 1

        counter_list = 0
        biggest_value = 0
        counter_element = counter_element + 1
    return array


def readCSVFile(filename: str) -> list:
    import csv
    data = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for i, row in enumerate(csvReader):
            if i == 0:
                data.append(row)  ## pass first row (feat names)
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
                    newRow.append(float(data[rowIndex][featureIndex]))
                except ValueError:
                    newRow.append(0.0)
        # Replace new row
        data[rowIndex] = newRow


def printFeatureIndcies(data):
    for i, val in enumerate(data[0]):
        print((i, val))


if __name__ == '__main__':
    exit(main())
