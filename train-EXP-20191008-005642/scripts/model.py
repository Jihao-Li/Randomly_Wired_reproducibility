import torch
import torch.nn as nn
import torch.nn.functional as F


class Triplet(nn.Module):
    """ReLU-Conv-BN"""
    def __init__(self, C_in, C_out, stride, in_degree, kernel_size=3, padding=1, affine=True):
        super(Triplet, self).__init__()

        self.in_degree = in_degree
        # For non-input node, weighted sum is needed
        if not self.in_degree == 0:
            self.arch_parm = nn.Parameter(torch.zeros(in_degree, requires_grad=True))

        self.rcb_unit = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, *inputs):
        if not self.in_degree == 0:
            return self.rcb_unit(sum(arch_parm * x for arch_parm, x in zip(torch.sigmoid(self.arch_parm), inputs)))
        else:
            return self.rcb_unit(inputs[0])


class StageOfRandomGraph(nn.Module):
    def __init__(self, graph_info_dict, C_in, C_out):
        super(StageOfRandomGraph, self).__init__()

        self.num_nodes = graph_info_dict["num_nodes"]
        self.in_degree = graph_info_dict["in_degree"]
        self.input_nodes = graph_info_dict["input_nodes"]
        self.output_nodes = graph_info_dict["output_nodes"]
        self.source_nodes = graph_info_dict["source_nodes"]

        self.nodes = nn.ModuleList()
        for node in range(self.num_nodes):
            if node in self.input_nodes:
                self.nodes.append(Triplet(C_in, C_out, stride=2, in_degree=self.in_degree[node]))
            else:
                self.nodes.append(Triplet(C_out, C_out, stride=1, in_degree=self.in_degree[node]))

    def forward(self, x):
        every_node_output = dict()         # the output of every node
        for node in range(self.num_nodes):
            if node in self.input_nodes:
                every_node_output[node] = self.nodes[node](x)
            else:
                every_node_output[node] = self.nodes[node](*[every_node_output[_node] for _node in self.source_nodes[node]])
        return sum([every_node_output[node] for node in self.output_nodes]) / len(self.output_nodes)


class RandomlyWiredNN(nn.Module):
    def __init__(self, base_channels, num_classes, output_channels, regime, graph_info_dicts):
        super(RandomlyWiredNN, self).__init__()

        self.base_channels = base_channels
        self.num_classes = num_classes
        self.output_channels = output_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.base_channels // 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.base_channels // 2),
        )

        # small regime or regular regime
        if regime:
            self.last_conv_channels = self.base_channels * 4
            self.conv2 = Triplet(self.base_channels // 2, self.base_channels, stride=2,
                                 padding=1, in_degree=0, kernel_size=3, affine=True)
            self.conv3 = StageOfRandomGraph(graph_info_dicts[0], self.base_channels, self.base_channels)
            self.conv4 = StageOfRandomGraph(graph_info_dicts[1], self.base_channels, self.base_channels * 2)
            self.conv5 = StageOfRandomGraph(graph_info_dicts[2], self.base_channels * 2, self.last_conv_channels)
        else:
            self.last_conv_channels = self.base_channels * 8
            self.conv2 = StageOfRandomGraph(graph_info_dicts[0], self.base_channels, self.base_channels)
            self.conv3 = StageOfRandomGraph(graph_info_dicts[1], self.base_channels, self.base_channels * 2)
            self.conv4 = StageOfRandomGraph(graph_info_dicts[2], self.base_channels * 2, self.base_channels * 4)
            self.conv5 = StageOfRandomGraph(graph_info_dicts[3], self.base_channels * 4, self.last_conv_channels)

        self.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.last_conv_channels, self.output_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.output_channels),
        )
        self.fc = nn.Linear(self.output_channels, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.last_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)       # flatten
        x = self.fc(x)

        return x


if __name__ == "__main__":
    import os
    import argparse
    from tensorboardX import SummaryWriter
    from graph.graph_libs import read_graph_info

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser("ImageNet")
    parser.add_argument('--regime', type=bool, default=True, help='small is True, regular is False')
    parser.add_argument('--base_channels', type=int, default=78, help='select in [78, 109, 154]')
    parser.add_argument('--output_channels', type=int, default=1280, help='1280 in original parper')
    parser.add_argument('--graph_txt', type=str, default='ws_4_075_conv', help='graph info txt')
    args = parser.parse_args()

    NUM_CLASSES = 1000
    graph_info_dict3 = read_graph_info(args.graph_txt + "3.txt")
    graph_info_dict4 = read_graph_info(args.graph_txt + "4.txt")
    graph_info_dict5 = read_graph_info(args.graph_txt + "5.txt")
    graph_info_dicts = [graph_info_dict3, graph_info_dict4, graph_info_dict5]
    model = RandomlyWiredNN(args.base_channels, NUM_CLASSES, args.output_channels, args.regime, graph_info_dicts)
    model = model.cuda()
    writer = SummaryWriter(log_dir="temp_test/")

    inputs = torch.rand(2, 3, 224, 224).cuda()
    writer.add_graph(model, (inputs,))

    output = model(inputs)
    print(output)
