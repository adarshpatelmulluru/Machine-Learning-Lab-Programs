import numpy as np
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

iris_data =load_iris()

data = iris_data.data[:5]

def proximity_matrix(data):
    n = data.shape[0]
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            matrix[i][j] = np.linalg.norm(data[i]-data[j])
            matrix[j][i] = matrix[i][j]
    return matrix

def plotting_dendrogram(data,linkage_method):
    linkage_matrix = linkage(data,method=linkage_method)
    dendrogram(linkage_matrix)
    plt.title(f"Dendogram for {linkage_method} method")
    plt.xlabel("Data points")
    plt.ylabel("Distance")
    plt.show()

correlation_matrix = np.corrcoef(data)
sns.heatmap(correlation_matrix,annot=True,cmap="viridis")
plt.show()

proximity_matrix = proximity_matrix(data)
print(proximity_matrix)

plotting_dendrogram(data,'single')

plotting_dendrogram(data,'complete')
