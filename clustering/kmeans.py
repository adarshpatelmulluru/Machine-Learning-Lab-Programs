import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris_data  = load_iris()
iris = iris_data.data

class Kmeans():
    def __init__(self,n_clusters = 3,max_iterations=100,tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations=max_iterations
        self.tolerance = tolerance

    def fit(self,x):
        self.centroids = x[:self.n_clusters]
        x_remaining = x[self.n_clusters:]

        for _ in range(self.max_iterations):
            self.labels = self.assign_clusters(x_remaining)
            new_centroids = self.calculate_centroids(x_remaining)

            if np.all(np.abs(new_centroids-self.centroids) < self.tolerance):
                break
        
            self.centroids = new_centroids

    def assign_clusters(self,x):
        distances = np.zeros((x.shape[0],self.n_clusters))
        for k in range(self.n_clusters):
            distances[:,k] = np.linalg.norm(x-self.centroids[k],axis=1)
        return np.argmin(distances,axis=1)
        
    def calculate_centroids(self,x):
        centroids = np.zeros((self.n_clusters,x.shape[1]))
        for k in range(self.n_clusters):
            centroids[k] = x[self.labels == k].mean(axis=0)
        return centroids
    
    def predict(self,x):
        return self.assign_clusters(x)
    
cluster = Kmeans(n_clusters=3)
cluster.fit(iris)
labels = cluster.predict(iris)
print(labels)

plt.figure(figsize=(10,6))
plt.scatter(iris[:,0],iris[:,1],c=labels,cmap='jet',marker='o')
plt.scatter(cluster.centroids[:,0],cluster.centroids[:,1],c='red',s=300,marker='x')
plt.xlabel(iris_data.feature_names[0])
plt.ylabel(iris_data.feature_names[1])
plt.title('K-means Clustering on Iris Dataset')
plt.show()
