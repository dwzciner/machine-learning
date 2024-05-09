import numpy as np
import os
import cv2

class NeuralNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim,lr):
        self.input_dim = input_dim #神经网络的输入层
        self.hidden_dim = hidden_dim #神经网络的输入层
        self.output_dim = output_dim #神经网络的输出层
        self.lr=lr                  #学习率
        # 初始化权重矩阵和偏置向量
        self.weights1 = 0.1*np.random.randn(self.input_dim, self.hidden_dim)
        self.bias1 = np.zeros((1, self.hidden_dim))

        self.weights2 = 0.1*np.random.randn(self.hidden_dim, self.output_dim)
        self.bias2 = np.zeros((1, self.output_dim))
        # print(self.weights1)
        # print(self.weights2)
    def sigmoid(self, x):  #sigmoid激活函数
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x): #sigmoid激活函数的导数
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self,x):  #最大软间隔模型
        if x.ndim == 2:  #处理多条数据
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)  # 防止溢出
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self,y, t): #交叉熵损失函数
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def forward(self, X):
        # MPL前向传播计算预测值

        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, X, y, output):
        # 反向传播训练网络，更新权重矩阵和偏置向量
        self.error=self.cross_entropy_error(output,y)
        #self.error = y - output
        self.deltz2 =(output-y)/X.shape[0];

        self.error_hidden_layer = np.dot(self.deltz2, self.weights2.T)
        ##
        self.deltz1 =  self.sigmoid_derivative(self.z1)*self.error_hidden_layer
        self.weights1 -= np.dot(X.T, self.deltz1)
        self.bias1 -= np.sum(self.deltz1, axis=0, keepdims=True)
        self.weights2 -= self.lr*np.dot(self.a1.T, self.deltz2)
        self.bias2 -=self.lr*np.sum(self.deltz2,axis=0)

    def train(self, X, y):  #训练模型
        output = self.forward(X)
        self.backward(X, y, output)

    def predict(self, X):  #预测
        return self.forward(X)

    def accuracy(self, x, t): #计算预测准确率
        y = self.predict(x)
        p = np.argmax(y, axis=1)
        q = np.argmax(t, axis=1)
        acc = np.sum(p == q) / len(y)
        return acc

def load_mnist(image_folder):
    images = []
    labels = []
    for image_file in os.listdir(image_folder):
        #if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
            parts = image_file.split('_')
            label = int(parts[-1].split('.')[0])  # 获取文件名中的标签
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))  # Resize the image to 28x28
            images.append(image.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

def normalize(X):       #数据归一化
    normalize_X=X/255;
    return normalize_X

def transferm_to_one_hot(Y): #将0-9的标签转化为独热编码
    T = np.zeros((Y.size, 10))
    for idx, row in enumerate(T):
        row[Y[idx]] = 1
    return T

if __name__ == '__main__':
    #预处理images label
    test_images,test_labels = load_mnist('test_images')
    train_images,train_labels = load_mnist('train_images')
    print(train_labels)
    # 数据归一化
    train_images_normalized = normalize(train_images)
    test_images_normalized = normalize(test_images)
    #print(train_images_normalized)
    # 标签转换为独热编码
    train_labels_one_hot = transferm_to_one_hot(train_labels)
    test_labels_one_hot = transferm_to_one_hot(test_labels)

    # 初始化神经网络
    mynet = NeuralNetwork(784, 100, 10, 0.5)

    epoch = 500  # 迭代次数

    # 训练过程
    train_loss_list = []
    train_acc_list = []
    validation_acc_list = []
    validation_loss_list = []

    for i in range(epoch):
        # 使用整个训练集进行训练
        x_batch = train_images_normalized
        y_batch = train_labels_one_hot

        # 训练模型
        mynet.train(x_batch, y_batch)

        # 计算损失和准确率
        train_loss = mynet.cross_entropy_error(mynet.predict(x_batch), y_batch)
        train_loss_list.append(train_loss)

        train_acc = mynet.accuracy(train_images_normalized, train_labels_one_hot)
        train_acc_list.append(train_acc)

        validation_loss = mynet.cross_entropy_error(mynet.predict(test_images_normalized), test_labels_one_hot)
        validation_loss_list.append(validation_loss)

        validation_acc = mynet.accuracy(test_images_normalized, test_labels_one_hot)
        validation_acc_list.append(validation_acc)
        if i % 10 == 0:
            print(f"Epoch {i}: Train Loss: {train_loss}, Train Acc: {train_acc}, Validation Loss: {validation_loss}, Validation Acc: {validation_acc}")