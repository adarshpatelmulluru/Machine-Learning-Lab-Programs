import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Multi-Layer Perceptron class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        error = y - output
        delta_output = error * sigmoid_derivative(output)
        
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0) * self.learning_rate
        self.weights_input_hidden += x.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * self.learning_rate

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, x):
        return int(self.forward(x)>0.5)


X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_not = np.array([[0], [0], [1], [0]])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

mlp_and_not = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_and_not.train(X_and_not, y_and_not)

print("AND-NOT Function:")
for xi in X_and_not:
    print(f"{xi} -> {mlp_and_not.predict(xi)}")

mlp_xor = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_xor.train(X_xor, y_xor)

print("\nXOR Function:")
for xi in X_xor:
    print(f"{xi} -> {mlp_xor.predict(xi)}")
