from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        # print("map=", fltLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


'''returns the value used for each leaf'''
def regLeaf(dataSet):
    # print("mean = ", dataSet[:, -1])
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


'''选择最好的数据集划分特征'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # print("chooseBestSplit=", dataSet)
    tolS = ops[0]  # error descent
    tolN = ops[1]  # min data
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    # print("featSet=", list(range(n-1)))
    for featIndex in range(n-1):
        # print("featIndex=", featIndex)
        # print("valueSet=", set(dataSet[:, featIndex].T.tolist()[0]))
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # print("splitVal=", splitVal)
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # print("mat0=", mat0)
            # print("mat1=", mat1)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                # print("featIndex=", featIndex, " splitVal=", splitVal,
                #       " shape1=", shape(mat0)[0], " shape2=", shape(mat1)[0])
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                # print("newS=", newS)
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # print("bestS=", bestS, " bestIndex=", bestIndex, "bestValue=", bestValue)
    if (S - bestS) < tolS:
        print("tolS=", tolS, "S-bestS=", S - bestS)
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        print("tolN=", tolN, " num0=", shape(mat0)[0], " num1=", shape(mat1)[0])
        return None, leafType(dataSet)
    return  bestIndex, bestValue


'''CART算法实现'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # print("createTree=", dataSet)
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val;
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


'''判断当前节点是否内部节点，
true：是内部节点，不是叶节点，
false：不是内部节点，是叶节点'''
def isTree(obj):
    return (type(obj).__name__=='dict')

'''对树进行塌陷处理'''
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

'''待剪枝的树与剪枝所需的测试数据'''
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 对测试数据进行切分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 左右均为叶节点，则可以执行合并操作
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    m, n = shape(dataSet)
    # 第一列为常量
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, '
                        'try increasing this second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


'''模型树的叶节点生成函数'''
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


'''在给定数据集上计算误差'''
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


'''对回归树叶节点进行预测'''
def regTreeEval(model, inDat):
    return float(model)

'''对模型树叶节点进行预测'''
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


'''用树回归进行预测'''
def treeForeCast(tree, inData, modelEval = regTreeEval):
    # 如果是叶节点
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            # 叶节点预测
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            # 叶节点预测
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


def test():
    myDat = loadDataSet('data/ex2.txt')
    myMat = mat(myDat)
    # print("test=", myMat)
    tree = createTree(myMat, ops=(0,1))
    print("tree=", tree)
    myDatTest = loadDataSet("data/ex2test.txt")
    myMat2Test = mat(myDatTest)
    pruneTree = prune(tree, myMat2Test)
    print("pruneTree", pruneTree)


def testModel():
    myMat = mat(loadDataSet("data/exp2.txt"))
    tree = createTree(myMat, modelLeaf, modelErr, (1, 10))
    print("tree=", tree)


def testTreeForecast():
    trainMat = mat(loadDataSet("data/biketrain.txt"))
    testMat = mat(loadDataSet("data/biketest.txt"))
    myTree = createTree(trainMat, ops=(1, 20))
    print("tree=", myTree)
    yHat = createForeCast(myTree, testMat[:, 0])
    coef = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print("regTree corrcoef=", coef)
    modelTree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat = createForeCast(modelTree, testMat[:, 0], modelTreeEval)
    print("modelTree corrcoef=", corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    ws, X, Y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0,0]
    print("linear corrcoef=", corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])


# test()
# testModel()
testTreeForecast()






















