import os
import math
import argparse
import numpy as np


def shuffle(n, edges):
    """
    shuffle the connection relationship
    :param n: he num of nodes
    :param edges: the num of edges
    :return: 
    """
    mapping = np.random.permutation(range(n))
    shuffled = list()
    for edge in edges:
        s, e = edge
        shuffled.append(sorted((mapping[s], mapping[e])))
    return sorted(shuffled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Erdos-Renyi graph generator')
    parser.add_argument('-n', '--n_nodes', type=int, default=32, help="number of nodes for random graph")
    parser.add_argument('-p', '--prob', type=float, default=0.2, help="probablity of node connection for ER")
    parser.add_argument('-o', '--out_txt', type=str, default='er_02.txt', help="name of output txt file")
    parser.add_argument('-s', '--seed', type=float, default=5, help="random seed")
    args = parser.parse_args()
    n, p = args.n_nodes, args.prob
    np.random.seed(args.seed)

    if p < math.log(n) / n:
        print("Warning: p is to small for given n.")
        print("This may make generated graph being disconnected.")

    edges = list()
    rand = np.random.uniform(0.0, 1.0, size=(n, n))

    # All node pais are connected according to probability
    for i in range(n):
        for j in range(i + 1, n):
            if rand[i][j] < p:
                edges.append((i, j))
    edges = shuffle(n, edges)

    os.makedirs('generated', exist_ok=True)
    with open(os.path.join('generated', args.out_txt), 'w') as f:
        f.write(str(n) + '\n')             # the num of nodes
        f.write(str(len(edges)) + '\n')    # the num of edges
        for edge in edges:
            f.write('%d %d\n' % (edge[0], edge[1]))
