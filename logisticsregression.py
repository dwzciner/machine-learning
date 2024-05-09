import re
import numpy as np

def loadData(fileName):
    x = []
    y = []
    regex = re.compile('\s+')
    with open(fileName, 'r') as f:
        readlines = f.readlines()
        for line in readlines:
            dataLine = regex.split(line)
            dataList = [float(x) for x in dataLine[0:-1]]
            xList = dataList[0:21]
            x.append(xList)
            y.append(dataList[-1])
    return x, y

def matrix(x):
    return np.matrix(x)

class LogiRegre():
    def __init__(self,alpha,times):
        self.alpha = alpha
        self.times = times

    def Sigmoid(self, x):
        indices_pos = np.nonzero(x >= 0)
        indices_neg = np.nonzero(x < 0)

        y = np.zeros_like(x)
        y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
        y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))

        return y

    def gradientdescent(self,X,Y):
        self.w_ = matrix(np.zeros(1 + X.shape[1]))
        self.loss = []

        for i in range(self.times):
            z = np.dot(self.w_[:,1:],X.T) + self.w_[:,0]
            #z = np.dot(self.w_[1:],X.T) + self.w_[0]
            #z = X * self.w_
            p = self.Sigmoid(z)
            cost = -np.sum(Y.T * np.log(p + 1e-6) + (1-Y).T * np.log(1-p + 1e-6)) / 299
            self.loss.append(cost)

            self.w_[:,0] += self.alpha * np.sum(Y - p) / 299
            for i in range(1,22):
                self.w_[:,i] += self.alpha * np.sum(np.dot(Y - p,X[:,i - 1])) / 299

    def predict_pro(self,X):
        z = np.dot(self.w_[:,1:],X.T) + self.w_[:,0]
        p = self.Sigmoid(z)

        #转换为二维列向量
        p = p.reshape(-1,1)

        return np.concatenate([1 - p,p],axis=1)

    def predict(self,X):
        return np.argmax(self.predict_pro(X),axis=1)

    def accuracy(self,X,Y):
        ac = 0
        for i in range(67):
            if(X[i] == Y[i]):
                ac += 1
        return ac/67 * 100

if __name__ == '__main__':
    alpha = 0.05
    times = 50000
    lr = LogiRegre(alpha,times)
    x_train,y_train = loadData('horseColicTraining.txt')
    x_test,y_test = loadData('horseColicTest.txt')

    x_train_m = matrix(x_train)
    y_train_m = matrix(y_train)
    x_test_m = matrix(x_test)
    y_test_m = matrix(y_test)


    lr.gradientdescent(x_train_m,y_train_m)
    print('训练模型参数')
    print(lr.w_)
    #print(lr.predict(x_test_m)[2])
    #print(int(y_test[0]))

    print()
    s = 'alpha:{}times:{}训练出的模型的准确率为{}%'.format(alpha,times,lr.accuracy(lr.predict(x_test_m),y_test))
    print(s)