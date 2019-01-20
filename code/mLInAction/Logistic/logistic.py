import math
import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("data/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat




'''梯度上升优化'''
def gradAscent(dataMatIn, classLabels, sigmoid):
    dataMatrix = np.mat(dataMatIn)
    # print("dataMatrix=", dataMatrix)
    labelMat = np.mat(classLabels).transpose()
    # print("labelMat=", labelMat)
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    print("initial weights=", weights)
    sigmoid = np.vectorize(sigmoid)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights


'''画出数据集和logistic回归最佳拟合直线的函数'''
def plotBestFit(wei):
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


'''随机梯度上升'''
def stocGradAscent0(dataMatrix, classLabels, sigmoid):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    sigmoid = np.vectorize(sigmoid)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


'''改进的随机梯度上升'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        if(j % 250 == 0):
            print("-------- iter=%d --------" % j)
        dataIndex = list(range(m))
        # print("dataIndex=", dataIndex)
        for i in range(m):
            # alpha decrease with iteration, does not go to 0 because of the constant
            alpha = 4/(1.0+j+i) + 0.05
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def sigmoid(inX):
    # print("inX=", inX)
    if inX > 709.0:
        inX = 709.00
    if inX < -709.00:
        inX = -709.00
    return 1.0/(1+math.exp(-inX))

'''logistic回归分类函数'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob> 0.5:
        return 1.0
    else:
        return 0.0

'''病马生死预测'''
def colicTest():
    frTrain = open('data/horseColicTraining.txt')
    frTest = open("data/horseColicTest.txt")
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))
    print("trainingSet shape", np.shape(trainingSet))
    print("trainingLabels shape", np.shape(trainingLabels))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 800)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    print("errorCount = %d, numTestVec=%d" % ( errorCount, numTestVec))
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        print("********** test = %d ***************" % k)
        errorSum += colicTest()
    print("after %d iterations the average error rate is %f "% (numTests, errorSum/float(numTests)))


dataArr, labelMat = loadDataSet()
# print(dataArr)
# print(labelMat)
# weight = gradAscent(dataArr, labelMat, sigmoid)
# print("gradAscent=", weight)
# plotBestFit(weight)

# weight = stocGradAscent0(np.array(dataArr), labelMat, sigmoid)
# print("stocGradAscent=", weight)
# plotBestFit(weight)

multiTest()



