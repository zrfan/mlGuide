from numpy import *
import matplotlib.pyplot as plt

'''数据导入函数'''


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


'''标准回归函数'''


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断行列式是否为0,如果行列式为0，计算逆矩阵的时候将出现错误
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


'''绘图'''


def plot(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print("xMat=", xMat[: 1].flatten())
    # print("yMat=", yMat.T[:, 0].flatten())
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
    print("corrcoef=", corrcoef(yHat.T, yMat))


'''局部加权线性回归函数'''


def lwlr(testPoint, xArr, yArr, k=1.0):
    # print("k=", k)
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角权重矩阵，为每个样本点初始化权重
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        # 随着样本点与待预测点距离的递增，权重值大小以指数级衰减
        # 参数k控制衰减的速度
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


'''给定x空间任意一点，计算出对应的预测值yHat'''


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotyHat(xArr, yArr, yHat):
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def testAbalone():
    abX, abY = loadDataSet("data/abalone.txt")
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print(rssError(abY[0:99], yHat01.T))
    print(rssError(abY[0:99], yHat1.T))
    print(rssError(abY[0:99], yHat10.T))
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print(rssError(abY[100:199], yHat01.T))
    print(rssError(abY[100:199], yHat1.T))
    print(rssError(abY[100:199], yHat10.T))


'''岭回归'''


def ridgeRegres(xMat, yMat, lam=0.2):
    # 给定lambda下的岭回归求解
    # 构建矩阵xTx
    xTx = xMat.T * xMat
    # lam乘以单位矩阵
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("the matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 为了使用岭回归和缩减技术，需要对特征做标准化处理，所有特征都减去各自的均值并除以方差
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        # 这里的lambda应以指数级变化，
        # 这样可以看出lambda在取非常小的值时和取非常大的值时分别对结果造成的影响
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def testRidge():
    abX, abY = loadDataSet("data/abalone.txt")
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


'''regularize by columns'''


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat


'''前向逐步线性回归'''


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    # 标准化
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def stageWiseTest():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xArr, yArr = loadDataSet("data/abalone.txt")
    # ws = stageWise(xArr, yArr, 0.01, 200)
    # print(ws)

    # ax.plot(ws)
    # plt.show()

    ws = stageWise(xArr, yArr, 0.005, 1000)
    print(ws)
    ax.plot(ws)
    plt.show()

    xMat = mat(xArr)
    xMat = regularize(xMat)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    weights = standRegres(xMat, yMat.T)
    print(weights.T)
    # ax.plot(weights)
    # plt.show()


from time import sleep
import json
import urllib.request

'''购物信息获取函数'''


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = "https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json" % (
        myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t %d\t %d\t %f\t %f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item %d" % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def testDataCollect():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)


'''交叉验证测试岭回归'''
'''numVal:交叉验证的次数'''
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        # 数据混洗
        random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # ridgeTest使用30个不同的lambda值创建了30组不同的回归系数
        # 使用上述测试集用30组回归系数来循环测试回归效果
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            # 岭回归需要使用标准化之后的数据，因此测试数据也需要用与测试集相同的参数来执行标准化
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX)
            varTrain = var(matTrainX)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            # errorMat中保存了ridgeTest里每个lambda值对应的多个误差值
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
        meanErrors = mean(errorMat, 0)
        minMean = float(min(meanErrors))
        bestWeights = wMat[nonzero(meanErrors==minMean)]
        xMat = mat(xArr)
        yMat = mat(yArr).T
        meanX = mean(xMat, 0)
        varX = var(xMat, 0)
        unReg = bestWeights/varX
        print("the best model from Ridge Regression is:\n", unReg)
        print("with constant term:", -1*sum(multiply(meanX, unReg)) + mean(yMat))

# xArr, yArr = loadDataSet("data/ex0.txt")
# ws = standRegres(dataArr, labelArr)
# print("ws=", ws)
# plot(dataArr, labelArr, ws)
# print("lwlr=", lwlr(xArr[0], xArr, yArr, 1.0), " yArr=", yArr[0])
# print("lwlr=", lwlr(xArr[0], xArr, yArr, 0.001), " yArr=", yArr[0])
# yHat = lwlrTest(xArr, xArr, yArr, 1.0)
# print("yHat=", yHat)
# plotyHat(xArr, yArr, yHat)

# yHat = lwlrTest(xArr, xArr, yArr, 0.01)
# print("yHat=", yHat)
# plotyHat(xArr, yArr, yHat)

# yHat = lwlrTest(xArr, xArr, yArr, 0.003)
# print("yHat=", yHat)
# plotyHat(xArr, yArr, yHat)

# testAbalone()

# testRidge()
# stageWiseTest()
testDataCollect()
