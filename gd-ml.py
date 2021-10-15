import numpy as np

def main():
    trainData = readCSVFile('train.csv')
    testData = readCSVFile('test.csv')
    cols = (1, 4, 17, 18, 19, 43, 44, 46, 62, 77) # Features in hw
    selectedColumns(testData, cols)
    selectedColumns(trainData, cols)
    trainData = np.array(trainData)
         
    return 0

def printFeatureIndcies(data):
    for i,val in enumerate(data[0]):
        print((i,val))

def readCSVFile(filename : str) -> list:
    import csv
    data = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for row in csvReader:
            data.append(row)
    return data

def selectedColumns(data:list, columns:tuple):
    # Remove data not included in study
    # Need Cols 1, 4, 17, 18, 19, 43, 44, 46, 62, 77
    # This index base includes ID col.
    for rowIndex in range(len(data)):
        newRow = list()
        for featureIndex in range(len(data[rowIndex])):
            if featureIndex in columns:
                newRow.append(data[rowIndex][featureIndex])
        # Replace new row
        data[rowIndex] = newRow



    pass
if __name__ == '__main__':
    exit(main())