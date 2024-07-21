import numpy as np

def step_function(x):
    return 1 if x>0 else 0

class Perceptron:
    def __init__(self,input_size,learning_rate=0.1,epochs = 10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs =epochs

    def predict(self,x):
        output = np.dot(self.weights,x)+self.bias
        return step_function(output)
    
    def train(self,x,y):
        for _ in range(self.epochs):
            for xi,yi in zip(x,y):
                y_pred = self.predict(xi)
                cost = yi-y_pred #cost is similar as loss/error
                self.weights += self.learning_rate*cost*xi
                self.bias += self.learning_rate*cost


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])
y_or = np.array([0,1,1,1])

#training for AND OR Boolean functions

perceptron_and = Perceptron(input_size=2)
perceptron_or = Perceptron(input_size=2)

perceptron_and.train(x,y_and)
print(f"AND Boolean Function")
for xi in x:
    print(f"{xi[0]} and {xi[1]} is {perceptron_and.predict(xi)}")

print()

perceptron_or.train(x,y_or)
print(f"OR Boolean Function")
for xi in x:
    print(f"{xi[0]} or {xi[1]} is {perceptron_or.predict(xi)}")
