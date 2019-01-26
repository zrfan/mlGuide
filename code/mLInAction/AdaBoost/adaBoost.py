import numpy as np
import math
import matplotlib.pyplot as plt


def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


'''通过阈值比较对数据进行分类'''
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 小于阈值为负
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 大于阈值为负
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


'''单层决策树生成函数'''
'''找到数据集上最佳的单层决策树'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float(math.inf)
    # 遍历特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # print("arr=", dataMatrix.min(axis=1))
        # print("rangeMin=", rangeMin)
        # print("rangeMax=", rangeMax)
        stepSize = (rangeMax - rangeMin)/numSteps
        # 遍历特征的所有取值
        for j in range(-1, int(numSteps)+1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                threshval = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshval, inequal)
                errArr = np.mat(np.ones((m, 1)))
                # 如果predictedVals中的值不等于labelMat中的真正类别标签值，那么errArr的相应值为1
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % (i, threshval, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshval
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

'''基于单层决策树的AdaBoost训练过程'''
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # 每个数据点的权重，D的所有元素之和是1.0，初始值为1/m
    D = np.mat(np.ones((m, 1))/m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5 * math.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:", classEst.T)
        # 降低正确分类数据的权重
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error:", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


'''AdaBoost分类函数'''
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,
                                 classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print("aggClassEst=", aggClassEst)
    return np.sign(aggClassEst)


'''自适应数据加载函数'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    print("numFeat=", numFeat)
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        # print("lineArr=", lineArr)
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


'''ROC曲线的绘制及AUC计算函数'''
def plotROC(predStrengths, classLabels):
    # 绘制光标的位置, 从1.0，1.0开始绘图，直到<0, 0>
    cur = (1.0, 1.0)
    # ySum 用于计算AUC的值
    ySum = 0.0
    # 通过数组过滤计算正例的数目,即在y坐标轴上的步进数目
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    # y轴上的步长是1.0/numPosClas
    yStep =  1/ float(numPosClas)
    # x轴上的步长1.0/反例的数目
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            # 每得到一个标签为1.0的类，则要沿着y轴方向下降一个步长，即不断降低真阳率
            delX = 0;
            delY = yStep
        else:
            # 对于其他类别的标签则是在x轴方向上倒退了一个步长（假阴率方向）
            delX = xStep
            delY = 0
            # 为了计算AUC，需要对多个小矩形的面积进行累加，这些小矩形的宽度是xStep
            # 因此对所有矩形的高度进行累加，再乘以xStep得到其总面积
            # 所有高度的和ySum随着x轴的每次移动而渐次增加
            ySum += cur[1]
        print("delX=", delX, " delY=", delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for AdaBoost Horse Colic Detection System')
    ax.axis([0,1, 0, 1])
    plt.show()
    # 高度之和乘以xStep
    print("the Area Under the Curve is:", ySum * xStep)




dataMat, dataLabels = loadSimpData()
# print("dataMat=", dataMat)
# print("dataLabels=", dataLabels)
dataMat, dataLabels = loadDataSet("data/horseColicTraining2.txt")
# print("dataMat=", dataMat)
# print("dataLabels=", dataLabels)

D = np.mat(np.ones((5, 1))/5)
# bestStump, minError, bestClasEst = buildStump(dataMat, dataLabels, D)
# print("bestStump=", bestStump)
# print("minError=", minError)
# print("bestClasEst=", bestClasEst)

classifierArray, aggClassEst = adaBoostTrainDS(dataMat, dataLabels, 10)
print("classifierArray=", classifierArray)

# print(adaClassify([0, 0], classifierArray))
plotROC(aggClassEst.T, dataLabels)
