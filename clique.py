
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

# Example graph adjacency matrix J 
J = np.array([
    [0, 0, 1, 0,1,0],
    [0, 0, 1, 1,1,1],
    [1, 1, 0, 1,1,0],
    [0, 1, 1, 0,1,0],
    [1,1,1,1,0,0],
    [0,1,0,0,0,0]
])

# Desired clique size
k = 4
A = 10 # Scaling constant for interaction term
B = 1  # Scaling constant for constraint term

# Number of nodes
n = len(J)

# Initialize QUBO matrix
Q = np.zeros((n, n))

# Populate QUBO matrix
for i in range(n):
    Q[i, i] = 2*(A - B)-2*A*k
    for j in range(i + 1, n):
      if j>i:
        Q[i, j] = 2* (A-B*J[i,j])


# Convert QUBO matrix to dictionary format
Q_dict = {(i, j): Q[i, j] for i in range(n) for j in range(i, n)}

# Define the sampler
sampler = EmbeddingComposite(DWaveSampler(token=token, solver=solver))

# Solve the problem
response = sampler.sample_qubo(Q_dict, num_reads=100)

# Get the best solution
best_solution = response.first.sample

# Extract the solution nodes
clique_nodes = [i for i in best_solution if best_solution[i] == 1]
print("Nodes in the clique:", clique_nodes)



#draw graph
G = nx.Graph()
for i in range(n):
    for j in range(n):
        if J[i, j] == 1:
            G.add_edge(i, j)

# Positions of all nodes
pos = nx.spring_layout(G)

# Drawing the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=20, font_weight='bold')

# Highlight the clique nodes
nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, node_color='orange')

plt.title("Graph with Highlighted Clique")
plt.show()