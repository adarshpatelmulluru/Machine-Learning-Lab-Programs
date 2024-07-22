import heapq

def bfs(graph,start,goal):
    priority_queue=[]
    heapq.heappush(priority_queue,(start))
    visited = set()
    parent = {start:None}
    cost ={start:0}

    while priority_queue:
        current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node is goal:
            break

        for neighbour,edge_cost in graph[current_node]:
            if neighbour not in visited:
                heapq.heappush(priority_queue,(neighbour))
                parent[neighbour] = current_node
                cost[neighbour] = cost[current_node]+edge_cost

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path,cost[goal]
    

graph={
    'A':[('B',2),('C',4),('F',5)],
    'B':[('C',4),('E',3)],
    'C':[('D',6),('G',1)],
    'D':[('F',1)],
    'E':[('G',5)],
    'F':[('G',4)],
    'G':[]
}
start = 'A'
goal = 'G'

path,cost = bfs(graph,start,goal)
print(f"BFS search path is {path}")
print(f"total cost is {cost}")

# 3D surface plot

import pandas as pd
import matplotlib.pyplot as plt
dataframe = pd.read_csv(r'../../datasets/glass.csv')
x = dataframe['Al']
y = dataframe['Na']
z = dataframe['Mg']
ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap="jet")
plt.title("surface plot")
plt.show()
