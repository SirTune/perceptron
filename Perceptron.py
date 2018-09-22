'''
########################################################################################################################
Assignment 1 - Perceptron Algorithm
Created by Jet-Tsyn Lee in PyCharm Community Edition 4.5.2
Created 14/02/18, last updated 22/02/18

Required:
	Python 3.6.1
	Numpy 1.13.1

Binary Perceptron neural network that import data to train and classify 3 classes of data and classify
Test data.

########################################################################################################################
'''

import numpy as np


# Perceptron Network
class Perceptron(object):

    # Constructor
    def __init__(self, eta=0.5, epoch=10):
        self.eta = eta      # Learning Rate
        self.epoch = epoch  # number of iterations

    #############################

    # Function to train weights in the perceptron object
    # data - dataset (b, xm,...,y)
    # lambd - lambda coefficient for l2 regularisation factor, active if not = zero
    def trainWeights(self, data, lambd=0.0):

        # Create Weight Array, (include bias, -1 remove y label)
        self.weiArr = np.zeros(data.shape[1] - 1)

        # Set decimal places for print
        np.set_printoptions(precision=5)

        # Loop for the number of epoch iterations
        for epoch in range(self.epoch):

            # Shuffle dataset every iteration to ensure random order to maximize accuracy
            np.random.shuffle(data)

            errCnt = 0  # Count number of errors in iteration

            # Calculate initial accuracy
            accuracy = self.accuracy(data)

            # Loop for each data item and update weights accordingly
            for i in range(len(data)):

                # Variables
                input = data[i][:-1]    # Input data row
                actual = data[i][-1]    # Actual y label
                predicted = self.predict(input)     # Predicted classification
                error = (actual - predicted)        # Error

                # Update bias weight
                self.weiArr[0] = self.weiArr[0] + (self.eta * error)

                # Update input weights
                for j in range(1, len(self.weiArr)):

                    # ~~~~~~~~~~~~~~  WEIGHT UPDATE RULE  ~~~~~~~~~~~~~~~~~
                    # REGULARIZATION - add 2*lambda*Wi to the weight update
                    if lambd != 0 and error != 0.0:
                        #W(k+1) = W(k) + eta(y-phi(z)) + 2*lambda*W(k)
                        self.weiArr[j] = self.weiArr[j] + (self.eta * error * data[i][j]) - (2 * lambd * self.weiArr[j])

                    # Perceptrion Update
                    else:
                        # Wj+(delta)Wj, eta(y-phi(z))
                        self.weiArr[j] = self.weiArr[j] + (self.eta * error * data[i][j])

                # Count error and print information
                if error != 0:
                    errCnt += 1

            # Print epoch details
            print("EPOCH", epoch+1, "- \tWeights:", self.weiArr, "\tAccuracy: %d/%d = %.2f%%, \tError Found: %d" % \
                  (accuracy, len(data), ((accuracy / float(len(data))) * 100), errCnt))

        return self

    # Calculate the accuracy of the dataset
    def accuracy(self, data, printRes=False):
        correct = 0.0

        # Loop all data rows
        for iRow in data:
            input = iRow[:-1]
            output = iRow[-1]   # Actual y Label

            # Get predicted classification
            pred = self.predict(input)

            # check prediction accuracy
            if pred == output:
                correct += 1.0

        if printRes == True:
            print("Accuracy: %d/%d = %.3f%%" % (correct, len(data), ((correct / float(len(data))) * 100)))

        # returns number of correct predictions
        return correct


    # Predict class label based on inputs
    def predict(self, inputs):

        threshold = 0.0
        activation = 0.0

        # sum weights and input (w^t*x)
        for i in range(len(inputs)):
            activation += float(inputs[i]) * self.weiArr[i]

        # Return class label
        if activation >= threshold:
            return 1.0
        else:
            return -1.0


########################################################################################################################

# Import Dataset and store in class
class ImportData(object):

    # CONSTRUCTOR
    # iClass - stores data only for the specified class, otherwise, store all information
    def __init__(self, fileName, iClass=""):
        with open(fileName, 'r') as f:
            # Set array variables
            self.data = []
            self.y = []

            # loop each line in file
            for line in f:

                # split word in row
                row = line.split(',')
                row.insert(0, 1)  # add bias = 1 to position zero

                # Copy only required class data
                if iClass == row[5].rstrip('\n') or iClass == "":
                    # set values in to array for dataset and y set, remove trailing '\n'
                    self.data.append(row[:-1])
                    self.y.append(row[5].rstrip('\n'))

        # Array format
        self.data = np.asarray(self.data)  # set dataset as numpy array
        self.data = self.data.astype('float')  # set elements to float
        self.y = np.asarray(self.y)  # set y set as numpy



#########  FUNCTIONS  ##########
# Combines two datasets and y labels to a single array to input into perceptron,
# Classifier set for biniary classification to set the required class labels
def combine(arr1, ySet1, arr2=[], ySet2=[], classifier=""):

    # Converts the specified class to the classification label, -1,1
    def classNo(input, classifier):
        if classifier != "":
            if input == classifier:
                return 1
            else:
                return -1
        else:
            return input

    dataTemp = []

    # combine dataset to a single array
    # Dataset 1
    for i, j in zip(arr1, ySet1):
        row = np.append(i, np.expand_dims(classNo(j, classifier), 1), axis=0)
        dataTemp.append(row)

    # Dataset 2
    for i, j in zip(arr2, ySet2):
        row = np.append(i, np.expand_dims(classNo(j, classifier), 1), axis=0)
        dataTemp.append(row)

    # convert to np array and shuffle
    dataTemp = np.asarray(dataTemp)
    np.random.shuffle(dataTemp)

    return dataTemp

# Calculate mean of a data set
def mean(dataSet, arrLen=4):
    meanArr = np.zeros(arrLen)

    for iMean in range(len(meanArr)):
        mSum = sum(dataSet[:, iMean + 1])
        mCnt = np.count_nonzero(dataSet[:, iMean + 1])
        meanArr[iMean] = mSum / mCnt

    return meanArr

# Calcuate Standard deviation
def standardDeviation(dataSet, arrLen=4):
    stdArr = np.zeros(arrLen)
    meanArr = mean(dataSet)

    for iStd in range(len(stdArr)):
        for jRow in dataSet:
            stdArr[iStd] += np.square((jRow[iStd + 1] - meanArr[iStd]))
        sCnt = np.count_nonzero(dataSet[:, iStd + 1]) - 1
        stdArr[iStd] = np.sqrt(stdArr[iStd] / sCnt)

    return stdArr


###############   MAIN   ###############
if __name__ == '__main__':

    # Import data
    trainPath = 'train.data'
    testPath = 'test.data'

    # All Train data
    trainFile = ImportData(trainPath)

    # Test Data
    testFile = ImportData(testPath)

    # Train data separated to classes
    c1 = ImportData(trainPath, 'class-1')
    c2 = ImportData(trainPath, 'class-2')
    c3 = ImportData(trainPath, 'class-3')

    # Test Data separated to classes
    test1 = ImportData(testPath, 'class-1')
    test2 = ImportData(testPath, 'class-2')
    test3 = ImportData(testPath, 'class-3')


    # =====  QUESTION 3  =====
    print("\n##############  CLASS DISCRININATION (Q3)  ##############")

    # Set arrays for loop to run class comparisons against class L vs R
    # 1v2, 2v3, 1v3

    # Label and Eta rates for each test
    clsLblL = ['class-1','class-2','class-1']
    clsLblR = ['class-2','class-3','class-3']
    etaRates = [0.5, 0.1, 0.5]      # eta rate for each test

    # Training Data
    clsTrainL = [c1,c2,c1]
    clsTrainR = [c2,c3,c3]

    # Test data
    clsTestL = [test1,test2,test1]
    clsTestR = [test2,test3,test3]

    # Loop each test for the different class comparisons
    for iCls in range(len(clsLblL)):
        print("\n\n==========  ", clsLblL[iCls].upper() ," (+1) vs ", clsLblR[iCls].upper()," (-1)  ==========")

        # Combine data sets
        trainData = combine(clsTrainL[iCls].data, clsTrainL[iCls].y, clsTrainR[iCls].data, clsTrainR[iCls].y, clsLblL[iCls])

        # Create perceptron class and train with data
        ppn = Perceptron(eta=etaRates[iCls], epoch=20)
        ppn.trainWeights(trainData)

        # Compare against Test Data
        print("\n==========  TEST DATA  ==========")
        testData = combine(clsTestL[iCls].data, clsTestL[iCls].y, clsTestR[iCls].data, clsTestR[iCls].y, clsLblL[iCls])
        ppn.accuracy(testData, True)


    # =====  QUESTION 4  =====
    print("\n\n##############  CLASS 1 & 2 CFEATURE COMPARISON (Q4)  ##############")

    # Calculate mean and standard deviation of each data set
    print("\nDataSet 1 - Mean:", mean(c1.data))
    print("DatsSet 1 - Standard Deviation:", standardDeviation(c1.data))
    print("\nDataSet 2 - Mean:", mean(c2.data))
    print("DatsSet 2 - Standard Deviation:", standardDeviation(c2.data))

    data1v2 = combine(c1.data, c1.y, c2.data, c2.y, 'class-1')
    print("\nDataSet 1+2 - Mean:", mean(data1v2))
    print("DataSet 1+2 - Standard Deviation:", standardDeviation(data1v2))
    print("\nDifference - Mean:", mean(c1.data) - mean(c2.data))
    print("Difference - Standard Deviation:", standardDeviation(c1.data) - standardDeviation(c2.data))




    # ###############  MULTICLASS PERCEPTRON  #################

    # =====  QUESTION 5  =====
    print("\n\n##############  MULTICLASS (Q5)  ##############")

    # Set up array for loop
    mcClsLbl = ["class-1","class-2","class-3"]
    etaRates = [0.5, 0.5, 0.5]

    # Loop each class
    for iCls in range(len(mcClsLbl)):
        print("\n==========  ",mcClsLbl[iCls].upper(),"  ==========")

        # Create dataset of all classes, setting only the specific class as label 1
        mcData = combine(trainFile.data, trainFile.y, classifier=mcClsLbl[iCls])

        # Create new perceptron with all datasets and train
        mcPPN = Perceptron(eta=etaRates[iCls], epoch=20)
        mcPPN.trainWeights(mcData)

        # Compare trained perceptron with test data
        print("\n==========  TEST DATA  ==========")
        mcTest = combine(testFile.data, testFile.y, classifier=mcClsLbl[iCls])
        mcPPN.accuracy(mcTest, True)


    #=====  QUESTION 6  =====
    print("\n\n##############  L2 REGULARISATION (Q6)  ##############")

    # Lambda coefficients for l2 regularization
    regCoff = [0.01, 0.1, 1.0, 10.0, 100]

    # Loop for the multiclass data
    for iCls in range(len(mcClsLbl)):
        print("\n==========  ",mcClsLbl[iCls].upper(),"  ==========")

        # Create dataset and test data
        mcData = combine(trainFile.data, trainFile.y, classifier=mcClsLbl[iCls])
        mcTest = combine(testFile.data, testFile.y, classifier=mcClsLbl[iCls])

        # Loop for each coefficient
        for iCoff in regCoff:
            print("\n~~~~~  COEFFICIENT = ", iCoff,"  ~~~~~")

            # Create perceptron and train with coefficient
            regPPn = Perceptron(eta=etaRates[iCls], epoch=20)
            regPPn.trainWeights(mcData, iCoff)

            # Compare against test data
            print("\n==========  TEST DATA  ==========")
            regPPn.accuracy(mcTest, True)
