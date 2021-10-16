import sys

import numpy
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def main():
    trainData = readCSVFile('train.csv')  # Contains price labels
    YTrain = readCSVFile('train.csv')
    testData = readCSVFile('test.csv')
    cols1 = [1, 4, 17, 18, 19, 43, 44, 46, 62, 77]  # Features selected
    selectedColumns(testData, cols1)
    selectedColumns(trainData, cols1)
    selectedColumns(YTrain, [80])
    # for row in trainData: print(row)
    trainData = toNumpyMat(trainData)
    testData = toNumpyMat(testData)
    yTrain = toNumpyMat(YTrain)
    lr = 0.00000001
    # for row in testData: print(row)
    # use Sum of Squared diff SSD for cost func
    # Goal is to fit line/s over the train data and get the minimum error (Cost function value) using test data.
    # We try to automate this process by changing lr and selecting the best fit.
    # Theta mat start as random. Cost func J(theta1, theta2, ...) or J(theta)
    # for gradiant  [hypo - (actual y)]
    loss = []
    theta = np.random.rand(len(cols1)+1)
    ones = np.ones(len(trainData))

    trainData = trainData.transpose()
    theta = theta
    trainData = np.vstack([ones, trainData]) # Now in shape and intercept ones
    print(trainData.shape)
    print(theta.shape)
    counter = 0
    while True:
        try:
            Y = hypLinear(trainData.transpose(), theta) # Hypothesis
            dJ = (trainData*(Y-YTrain).transpose())/trainData.shape[1] # Gradiant
            theta = theta - (lr*dJ.transpose()) # Update theta
            currLoss = ((Y-YTrain)**2).sum()
            counter += 1
            if counter % 100 == 0: print('i: %d, loss = %d, theta: %s' % (counter, currLoss, theta[0]))
        except KeyboardInterrupt: break
    return 0


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
