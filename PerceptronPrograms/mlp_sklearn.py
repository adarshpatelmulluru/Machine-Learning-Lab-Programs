import numpy as np
from sklearn.neural_network import MLPClassifier

x_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_not = np.array([0, 0, 1, 0])

x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

mlp_and_not = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=10000,random_state=17)
mlp_and_not.fit(x_and_not, y_and_not)

y_pred_not_and = mlp_and_not.predict(x_and_not)
for xi,yi in zip(x_and_not,y_pred_not_and):
    print(f" {xi[0]} AND NOT{xi[1]} : {yi}")

mlp_xor = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=10000, random_state=17)
mlp_xor.fit(x_xor, y_xor)

y_pred_xor = mlp_xor.predict(x_xor)
for xi,yi in zip(x_xor,y_pred_xor):
    print(f" {xi[0]} XOR {xi[1]} : {yi}")
