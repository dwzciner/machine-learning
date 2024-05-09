import numpy as np
import matplotlib.pyplot as plt


def LoadDataSet(filename):
    X = [[], []]
    Y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            lineDataList = line.split('\t')
            lineDataList = [float(x) for x in lineDataList]
            X[0].append(lineDataList[0])
            X[1].append(lineDataList[1])
            Y.append(lineDataList[2])
    return X, Y

def mat(x):
    return np.matrix(np.array(x)).T

def DisplayData(X,Y,theta,k,b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat(X)[:, 1].flatten().A[0], mat(Y)[:, 0].flatten().A[0])
    x = np.linspace(0, 1, 50)
    y = x * theta[1,0] + theta[0,0]
    plt.plot(x,y,'r')

    y_g = x * k + b
    plt.plot(x,y_g,'y')
    plt.show()

def normalEquation(X,Y):
    A = mat(X).T.dot(mat(X))
    isInverse = np.linalg.det(A)
    if isInverse == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    else:
        I = A.I
        theta = I * mat(X).T * mat(Y)
        print(theta)
        return theta

def gradient_descent(alpha,times,x,y):
    w_ = [[0],[0]]
    for i in range(times):
        y_hat = x.dot(w_)
        error = y - y_hat
        #J = np.sum(error.dot(error.T)) / (2 * 200)

        w_[0] = w_[0] + alpha * np.sum(error)
        w_[1] = w_[1] + alpha * np.sum(error.T.dot(x))


    return w_


if __name__ == '__main__':
    X,Y = LoadDataSet('ex0.txt')
    x = np.asarray(X)
    y = np.asarray(Y)
    theta = normalEquation(X,Y)
    theta_1 = theta[0,0]
    theta_2 = theta[1,0]
    #print(theta)
    #DisplayData(X,Y,theta)

    #print(gradient_descent(0,0,100,np.asarray(X[1]),np.asarray(Y)))
    w= gradient_descent(0.001,1000,mat(X),mat(Y))
    print(w)
    DisplayData(X,Y,theta,w[1],w[0])


