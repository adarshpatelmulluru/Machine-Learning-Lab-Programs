import numpy as np
from sklearn.neural_network import MLPClassifier

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

mlp_and = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=1000,random_state=42)
mlp_and.fit(X_and, y_and)

y_pred_and = mlp_and.predict(X_and)
for xi,yi in zip(X_and,y_pred_and):
    print(f" {xi[0]} AND {xi[1]} : {yi}")

mlp_or = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_or.fit(X_or, y_or)

y_pred_or = mlp_or.predict(X_or)
for xi,yi in zip(X_or,y_pred_or):
    print(f" {xi[0]} OR {xi[1]} : {yi}")
