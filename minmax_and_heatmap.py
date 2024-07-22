def minmax(depth,node_index,maximizing_agent,values,path):
    if depth==3:
        return values[node_index],path+[node_index]
    
    if maximizing_agent:
        best = float("-inf")
        best_path=[]
        for i in range(2):
            value,new_path = minmax(depth+1,node_index*2+1,False,values,path+[node_index])
            if value > best:
                best = value
                best_path = new_path
            return best,best_path
    
    else:
        best = float("inf")
        best_path=[]
        for i in range(2):
            value,new_path = minmax(depth+1,node_index*2+1,True,values,path+[node_index])
            if value < best:
                best = value
                best_path = new_path
            return best,best_path
    
values = [2,5,6,1,5,7,8,12,25]

optimal_value,optimal_path = minmax(0,0,True,values,[])
print(f"optimal value is {optimal_value}")
print(f"optimal path is {optimal_path}")

#heat map

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv(r'../../datasets/iris.csv')
dataframe.drop('species',axis=1,inplace=True)
correlation_matrix = dataframe.corr()
sns.heatmap(correlation_matrix,cmap="viridis",annot=True)
plt.show()
