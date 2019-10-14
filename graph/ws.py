import os
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Watts-Strogatz graph generator')
    parser.add_argument('-n', '--n_nodes', type=int, default=32, help="number of nodes for random graph")
    parser.add_argument('-k', '--k_neighbors', type=int, default=4, help="connecting neighboring nodes for WS")
    parser.add_argument('-p', '--prob', type=float, default=0.75, help="probablity of rewiring for WS")
    parser.add_argument('-o', '--out_txt', type=str, default='ws_4_075_conv5.txt', help="name of output txt file")
    parser.add_argument('-s', '--seed', type=float, default=5, help="random seed")
    args = parser.parse_args()
    n, k, p = args.n_nodes, args.k_neighbors, args.prob
    np.random.seed(args.seed)

    assert k % 2 == 0, "k must be even."
    assert 0 < k < n, "k must be larger than 0 and smaller than n."

    adj = [[False] * n for _ in range(n)]       # adjacency matrix
    for i in range(n):
        adj[i][i] = True

    # initial connection，connect k//2 nodes on both sides
    for i in range(n):
        for j in range(i - k//2, i + k//2+1):
            real_j = j % n
            if real_j == i:
                continue
            adj[real_j][i] = adj[i][real_j] = True

    rand = np.random.uniform(0.0, 1.0, size=(n, k//2))
    for i in range(n):
        # repeat k//2
        for j in range(1, k//2 + 1):       # 'j' here is 'i' of paper's notation
            current = (i + j) % n      # select k//2 nodes to disconnect clockwise
            if rand[i][j - 1] < p:     # rewire
                unoccupied = [x for x in range(n) if not adj[i][x]]
                rewired = np.random.choice(unoccupied)       # select one unconnected node to be rewired
                adj[i][current] = adj[current][i] = False       # disconnect
                adj[i][rewired] = adj[rewired][i] = True        # rewire

    edges = list()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    edges.sort()

    os.makedirs('generated', exist_ok=True)
    with open(os.path.join('generated', args.out_txt), 'w') as f:
        f.write(str(n) + '\n')             # the num of nodes
        f.write(str(len(edges)) + '\n')    # the num of edges
        for edge in edges:
            f.write('%d %d\n' % (edge[0], edge[1]))
