import operator
from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing','flippers']
 #change to discrete values
    return dataSet, labels



def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):  # 参数：待划分数据集、划分数据集的特征、需返回的特征的值
    """按照给定特征划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分"""
    numFeatures = len(dataSet[0]) - 1  # 特征总个数
    baseEntropy = calShannonEnt(dataSet)  # 计算数据集的信息熵
    bestInfoGain, bestFeature = 0.0, -1  # 最优的信息增益值, 最优的Feature编号
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取各实例第i+1个特征
        uniqueVals = set(featList)
        newEntropy = 0.0  # 创建一个新的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        # 比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """多数表决方法决定叶子节点分类"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """创建决策树"""
    classList = [example[-1] for example in dataSet]
    # count()统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优列，得到最优列对应的label含义
    bestFeatLabel = labels[bestFeat]  # 获取label的名称
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]  # 取出最优列并对它的branch做分类
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    #print(myTree)
    return myTree
