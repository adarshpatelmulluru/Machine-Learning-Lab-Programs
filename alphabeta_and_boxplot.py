def alphabeta(depth,node_index,maximizing_agent,values,alpha,beta,path):
    if depth==3:
        return values[node_index],path+[node_index]
    
    if maximizing_agent:
        best = float("-inf")
        best_path=[]
        for i in range(2):
            value,new_path = alphabeta(depth+1,node_index*2+1,False,values,alpha,beta,path+[node_index])
            if value > best:
                best = value
                best_path = new_path
            alpha = max(alpha,value)
            if alpha>=beta:
                break
        return best,best_path
    else:
        best = float("inf")
        best_path=[]
        for i in range(2):
            value,new_path = alphabeta(depth+1,node_index*2+1,True,values,alpha,beta,path+[node_index])
            if value < best:
                best = value
                best_path = new_path
            beta = min(alpha,value)
            if alpha>=beta:
                break
        return best,best_path
    
values = [2,5,6,1,5,7,8,12,25]

optimal_value,optimal_path = alphabeta(0,0,True,values,float("-inf"),float("inf"),[])
print(f"optimal value is {optimal_value}")
print(f"optimal path is {optimal_path}")

#Box plot

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

dataset= sns.load_dataset('iris')
plt.figure(figsize=(10,6))
sns.boxplot(x=dataset['sepal_length'],y=dataset['sepal_width'])
plt.show()
