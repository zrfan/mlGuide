from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadData(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append(list(map(float, curLine)))
    dataMat = mat(dataMat)
    return dataMat

'''欧氏距离'''
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

'''k个随机质心集合'''
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

'''K均值聚类算法'''
def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print("centroids=", centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


'''二分K-均值聚类'''
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit=%f, and notSplit=%f"% (sseSplit, sseNotSplit))
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is:", bestCentToSplit)
        print("the len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    print("centList=", centList)
    return mat(centList), clusterAssment

'''球面距离计算'''
def disSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi/180) * sin(vecB[0, 1]*pi/180)
    b = cos(vecA[0, 1] * pi/180) * \
        cos(vecB[0, 1]*pi/180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0])/180)
    return arccos(a + b) * 6371.0


'''簇绘图函数'''
def clusterClubs(numClust=5):
    dataList = []
    for line in open("data/places.txt"):
        lineArr = line.split("\t")
        dataList.append([float(lineArr[4]), float(lineArr[3])])
    dataMat = mat(dataList)
    myCentroids, clustAssing = biKmeans(dataMat, numClust, distMeas=disSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarker = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgp = plt.imread('data/Portland.png')
    ax0.imshow(imgp)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = dataMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarker[i % len(scatterMarker)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], \
                    ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    print(myCentroids[:, 0])
    ax1.scatter(myCentroids[:, 0].flatten().A[0], \
                myCentroids[:, 1].flatten().A[0], marker='+',  s=300)
    plt.show()


def test():
    dataMat = loadData("data/testSet.txt")
    myCentroids, clustAssing = kMeans(dataMat, 4)
    print("myCentroids=", myCentroids)
    # print("clustAssing=", clustAssing)


def testBiKMeans():
    dataMat = loadData("data/testSet2.txt")
    centList, myAssment = biKmeans(dataMat, 3)
    print("centList=", centList)
    print("myAssment=", myAssment)


# test()
# testBiKMeans()
clusterClubs()
