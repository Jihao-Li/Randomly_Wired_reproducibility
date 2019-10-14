import os


def read_graph_info(graph_txt):
    graph_txt_graph = os.path.join("graph", "generated", graph_txt)
    # graph_txt_graph = os.path.join("generated", graph_txt)

    with open(graph_txt_graph, "r") as f:
        num_nodes = int(f.readline().strip())
        num_edges = int(f.readline().strip())
        in_degree = [0 for _ in range(num_nodes)]
        out_degree = [0 for _ in range(num_nodes)]
        source_nodes = [[] for _ in range(num_nodes)]

        edges = list()
        for _ in range(num_edges):
            s, e = map(int, f.readline().strip().split())
            edges.append((s, e))
            out_degree[s] += 1
            in_degree[e] += 1
            source_nodes[e].append(s)

        input_nodes = [node for node in range(num_nodes) if in_degree[node] == 0]
        output_nodes = [node for node in range(num_nodes) if out_degree[node] == 0]

        graph_info_dict = dict()
        graph_info_dict["num_nodes"] = num_nodes
        graph_info_dict["num_edges"] = num_edges
        graph_info_dict["edges"] = edges
        graph_info_dict["in_degree"] = in_degree
        graph_info_dict["out_degree"] = out_degree
        graph_info_dict["input_nodes"] = input_nodes
        graph_info_dict["output_nodes"] = output_nodes
        graph_info_dict["source_nodes"] = source_nodes
        return graph_info_dict


if __name__ == "__main__":
    graph_txt = "ws_4_075_conv3.txt"
    graph_info_dict = read_graph_info(graph_txt)
    print()
