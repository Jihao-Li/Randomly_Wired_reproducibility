import os
import torch
import shutil
import numpy as np
import torch.nn as nn


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def accuracy(output, target, topk=(1,)):
    """
    top1 and top5 acc
    :param output: 
    :param target: 
    :param topk: 
    :return: 
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)      # 返回最大的前maxk个元素和下标
    pred = pred.t()       # 将pred转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))     # 矩阵扩展

    res = []
    # topk中只有1或5，所以这里是计算top1或top5正确率
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))         # 计算准确率
    return res


def count_parameters_in_MB(model):
    """
    
    :param model: 
    :return:
    """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def model_param_flops_in_MB(model=None, input_res=[224, 224], multiply_adds=True):
    """
    It's better to set multiply_adds to False
    :param model: 
    :param input_res: row and col
    :param multiply_adds: True, multiply_adds is two FLOPs；False, multiply_adds is one FLOPs
    :return: 
    """
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1 = []
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.rand(3, input_res[1], input_res[0]).unsqueeze(0)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    # print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    return total_flops / 1e6


def save_parm(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_model(model, model_path):
    torch.save(model, model_path)


def load_model(model, model_path):
    return model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    """
    
    :param path: 
    :param scripts_to_save: 
    :return: 
    """
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def gen_mean_std(dataset, ratio):
    """
    calculate mean and std on channel dimension
    :param dataset: 
    :param ratio: 
    :return: 
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=True, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


if __name__ == "__main__":
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar10 = dset.CIFAR10(root='/dataset/cifar10/', train=True, download=True, transform=transforms.ToTensor())
    cifar10_mean, cifar10_std = gen_mean_std(cifar10, ratio=1.0)
    print(cifar10_mean, cifar10_std)

    # cifar100 = dset.CIFAR100(root='/dataset/cifar100/', train=True, download=True,
    #                          transform=transforms.Compose([transforms.ToTensor()]))
    # cifar100_mean, cifar100_std = gen_mean_std(cifar100, ratio=1.0)
    # print(cifar100_mean, cifar100_std)

    import os
    import argparse
    from model import RandomlyWiredNN
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
    # model = model.cuda()
    flops = model_param_flops_in_MB(model, multiply_adds=False)
    print()
