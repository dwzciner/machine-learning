import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y, learning_rate)
            loss = self.loss(y, predictions)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-10))
if __name__ == '__main__':
    # 假设X是输入数据，y是标签数据
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, (100, 1))

    input_size = X.shape[1]
    hidden_size = 32
    output_size = 1

    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.train(X, y, epochs=1000, learning_rate=0.001)
