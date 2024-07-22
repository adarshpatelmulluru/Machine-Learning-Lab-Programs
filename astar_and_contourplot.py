import heapq

def astar(graph,start,goal,heuristic):
    priority_queue = []
    heapq.heappush(priority_queue,(start,heuristic[start]))
    visited = set()
    g_cost = {start:0}
    parent = {start:None}

    while priority_queue:

        current_node,_ = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node is goal:
            break

        for neighbour,edge_cost in graph[current_node]:
            new_cost = g_cost[current_node] + edge_cost
            if neighbour not in g_cost or new_cost < g_cost[neighbour]:
                g_cost[neighbour] = new_cost
                f_cost = new_cost + heuristic[neighbour]
                heapq.heappush(priority_queue,(neighbour,f_cost))
                parent[neighbour] = current_node
        
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path,g_cost.get(goal,0)

graph={
    'A':[('B',2),('C',9),('F',5)],
    'B':[('C',4),('E',3)],
    'C':[('D',6),('G',1)],
    'D':[('F',1)],
    'E':[('G',5)],
    'F':[('G',4)],
    'G':[]
}
heuristic ={
    'A':2,
    'B':3,
    'C':1,
    'D':2,
    'E':1,
    'F':2,
    'G':0
}
start = 'A'
goal = 'G'

path,cost = astar(graph,start,goal,heuristic)
print(f"Astar patth is {path}")
print(f"Total cost is {cost}")

# Contour plot

import pandas as pd
import matplotlib.pyplot as plt
dataframe = pd.read_csv(r'../../datasets/glass.csv')
x = dataframe['Si']
y = dataframe['Na']
z = dataframe['Ca']
plt.tricontourf(x,y,z,levels =10,cmap ="viridis")
plt.show()
