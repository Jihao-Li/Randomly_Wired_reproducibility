import networkx as nx

n = 7
k = 4
p = 0.3
ws = nx.random_graphs.watts_strogatz_graph(n, k, p, seed=100)
nx.write_yaml(ws, "ws.yaml")
graph = nx.read_yaml("ws.yaml")
print()
