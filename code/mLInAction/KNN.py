'''
kNN: k Nearest Neighbors
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
        example:
        inX: [1.4, 1.6, 0.1]
        dataSet:
            [1.0, 1.1, 0.8],
            [1.0, 1.2, 1.3],
            [0,   0.1, 0.5],
            [0.3, 1.2, 0],
            [0.2, 0.2, 0]
Output:     the most popular class label
'''
import numpy as np
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]   # 行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    '''
    数组复制
    diffMat = [ [1.4, 1.6, 0.1],
                [1.4, 1.6, 0.1],
                [1.4, 1.6, 0.1],
                [1.4, 1.6, 0.1]
                [1.4, 1.6, 0.1]
        ]
    减去dataSet
    diffMat = [ [0.4, 0.5, -0.7],
                [0.4, 0.4, -1.2],
                [1.4, 1.5, -0.4],
                [1.1, 0.4, 0.1],
                [1.2, 1.4, 0.1]
        ]
    '''
    sqDiffMat = diffMat**2
    '''
    数组平方
    sqDiffMat = [ [0.16, 0.25, 0.49],
                [0.16, 0.16, 1.44],
                [1.96, 2.25, 0.16],
                [1.21, 0.16, 0.01],
                [1.44, 1.96, 0.01]
        ]
    '''
    sqDistances = sqDiffMat.sum(axis=1)  # 列累加
    '''
    列累加
    sqDistances = [ 1,
                    1.76,
                    4.37,
                    1.38,
                    3.41
    ]
    '''
    distances = sqDistances**0.5
    '''
    开方
    distances = [  1,
                    1.3, 
                    2.1,
                    1.11,
                    1.8
    ]
    '''
    sortedDistIndicies = distances.argsort()
    '''
    从小到大的索引顺序
    sortedDistIndicies = [0, 3, 1, 2]
    '''
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''分类器针对约会网站的测试代码'''
def datingClassTest():
    hoRatio = 0.05
    datingDataMat, datingLabels = knn.file2matrix("data/datingTestSet.txt")
    normMat, ranges ,minVals = knn.autoNorm(datingDataMat)
    m = normMat.shape[0]
    print("normMat shape:", normMat.shape)
    numTestVecs = int(m * hoRatio)
    print("numTestVecs=", numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knn.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(classifierResult)
        print("the classifier came back with : %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("the total error rate is : %f" % (errorCount/float(numTestVecs)), "error count=", errorCount)


'''约会网站预测函数'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDateMat, datingLabels = file2matrix("data/datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDateMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifyResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("you will propbably like this person: ", resultList[classifyResult - 1])


'''生成数据'''
def createDataSet():
    group = np.array([[1.0, 1.1, 0.8],
                      [1.0, 1.2, 1.3],
                      [0,   0.1, 0.5],
                      [0.3, 1.2, 0],
                      [0.2, 0.2, 0]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''从文本文件中解析数据'''
def file2matrix(filename):
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    print("dataSet shape=", returnMat.shape, " labels size=", len(classLabelVector))
    print(classLabelVector)
    return returnMat, classLabelVector


'''归一化'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals



'''图像2向量'''
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # print(lineStr)
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


'''手写数字识别系统测试'''
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("data/digits/trainingDigits/")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("data/digits/trainingDigits/%s" % fileNameStr)
    testFileList = listdir("data/digits/testDigits/")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("data/digits/testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        if classifierResult != classNumStr:
            print("the classfier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr))
            errorCount += 1.0

    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is : %f" % (errorCount/float(mTest)))


if __name__ == "__main__":
    matrix, labels = createDataSet()
    inX = [1.4, 1.6, 0.1]
    print("matrix=:", matrix)
    print("labels=:", labels)
    print("inX label=", classify0(inX, dataSet=matrix, labels=labels, k=1))









