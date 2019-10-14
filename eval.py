import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model import RandomlyWiredNN
from dataset import preprocess_imagenet
from graph.graph_libs import read_graph_info


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--data', type=str, default='/dataset/extract_ILSVRC2012', help='location of the data corpus')
parser.add_argument('--regime', type=bool, default=True, help='small is True, regular is False')
parser.add_argument('--batch_size', type=int, default=640, help='batch size')
parser.add_argument('--base_channels', type=int, default=78, help='select in [78, 109, 154]')
parser.add_argument('--output_channels', type=int, default=1280, help='1280 in original parper')
parser.add_argument('--train_flag', type=bool, default=False, help='train or test')
parser.add_argument('--model_path', type=str, default='./train-EXP-20191008-005642/model_top1.pt',
                    help='path of pretrained model')
parser.add_argument('--gpu', type=str, default="0, 1, 2, 3", help='gpu device id')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--graph_txt', type=str, default='ws_4_075_conv', help='graph info txt')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

NUM_CLASSES = 1000


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info('small regime or regular regime = %s' % args.regime)
    logging.info("args = %s", args)

    # read graph info
    if args.regime:
        graph_info_dict3 = read_graph_info(args.graph_txt + "3.txt")
        graph_info_dict4 = read_graph_info(args.graph_txt + "4.txt")
        graph_info_dict5 = read_graph_info(args.graph_txt + "5.txt")
        graph_info_dicts = [graph_info_dict3, graph_info_dict4, graph_info_dict5]
    else:
        graph_info_dict2 = read_graph_info(args.graph_txt + "2.txt")
        graph_info_dict3 = read_graph_info(args.graph_txt + "3.txt")
        graph_info_dict4 = read_graph_info(args.graph_txt + "4.txt")
        graph_info_dict5 = read_graph_info(args.graph_txt + "5.txt")
        graph_info_dicts = [graph_info_dict2, graph_info_dict3, graph_info_dict4, graph_info_dict5]

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = RandomlyWiredNN(args.base_channels, NUM_CLASSES, args.output_channels, args.regime, graph_info_dicts)

    # When training with multi GPUs, first execute nn.DataParallel() and then load model in eval stage
    if args.parallel:
        model = nn.DataParallel(model).cuda()
        print("multi GPUs")
    else:
        model = model.cuda()
        print("single GPU")
    utils.load_model(model, args.model_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("FLOPs = %fGB", utils.model_param_flops_in_MB(model, multiply_adds=False))

    _, test_queue = preprocess_imagenet(args)

    valid_acc, valid_obj, valid_speed = infer(test_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    logging.info('valid_obj %f', valid_obj)
    logging.info('valid_speed_per_image %f', valid_speed)


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    speed = utils.AvgrageMeter()

    model.eval()
    for step, (inputs, targets) in enumerate(valid_queue):
        n = inputs.size(0)
        inputs = inputs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            tic = time.time()
            logits = model(inputs)
            toc = time.time()
        val_loss = criterion(logits, targets)

        per_image_speed = 1.0 * (toc - tic) / n
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        objs.update(val_loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        speed.update(per_image_speed, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f %f', step, objs.avg, top1.avg, top5.avg, speed.avg)

    return top1.avg, objs.avg, speed.avg


if __name__ == "__main__":
    main()
