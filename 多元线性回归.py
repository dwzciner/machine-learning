import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import SGDRegressor


# 将数据转化成为矩阵
def matrix(x):
    return np.matrix(np.array(x)).T


# 加载数据
def loadData(fileName):
    x = []
    y = []
    regex = re.compile('\s+')
    with open(fileName, 'r') as f:
        readlines = f.readlines()
        for line in readlines:
            dataLine = regex.split(line)
            dataList = [float(x) for x in dataLine[0:-1]]
            xList = dataList[0:8]
            x.append(xList)
            y.append(dataList[-1])
    return x, y


# 求解回归的参数
def normalEquation(xmat, ymat):
    temp = xmat.T.dot(xmat)
    isInverse = np.linalg.det(xmat.T.dot(xmat))
    if isInverse == 0.0:
        print('不可逆矩阵')
        return None
    else:
        inv = temp.I
        return inv.dot(xmat.T).dot(ymat)


# 梯度下降求参数
def gradientDecent(alpha,times,x, y):
    w = matrix(np.zeros((1,8),dtype=float))
    for i in range(times):
        y_hat = x.dot(w)
        error = y - y_hat
        w[0] += alpha * np.sum(error.T) / len(xmat)
        w[1] += alpha * np.sum(error.T.dot(x[:,1])) / len(xmat)
        w[2] += alpha * np.sum(error.T.dot(x[:,2])) / len(xmat)
        w[3] += alpha * np.sum(error.T.dot(x[:,3])) / len(xmat)
        w[4] += alpha * np.sum(error.T.dot(x[:,4])) / len(xmat)
        w[5] += alpha * np.sum(error.T.dot(x[:,5])) / len(xmat)
        w[6] += alpha * np.sum(error.T.dot(x[:,6])) / len(xmat)
        w[7] += alpha * np.sum(error.T.dot(x[:,7])) / len(xmat)

    return w

if __name__ == "__main__":
    x, y = loadData('abalone1.txt')
    xmat = matrix(x).T
    ymat = matrix(y)
    # 通过equation来计算模型的参数
    theta = normalEquation(xmat, ymat)
    print('通过equation来计算模型的参数')
    print(theta)
    print("梯度下降")
    gtheta = gradientDecent(0.1,100000,xmat, ymat)
    print(gtheta)

