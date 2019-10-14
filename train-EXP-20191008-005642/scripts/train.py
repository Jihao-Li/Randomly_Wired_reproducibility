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
from tensorboardX import SummaryWriter
from dataset import preprocess_imagenet
from graph.graph_libs import read_graph_info


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--data', type=str, default='/dataset/extract_ILSVRC2012', help='location of the data corpus')
parser.add_argument('--regime', type=bool, default=True, help='small is True, regular is False')
parser.add_argument('--batch_size', type=int, default=640, help='batch size')
parser.add_argument('--base_channels', type=int, default=78, help='select in [78, 109, 154]')
parser.add_argument('--output_channels', type=int, default=1280, help='1280 in original parper')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--train_flag', type=bool, default=True, help='train or test')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--gpu', type=str, default="0, 1, 2, 3", help='gpu device id')
parser.add_argument('--graph_txt', type=str, default='ws_4_075_conv', help='graph info txt')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
args = parser.parse_args()

args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

NUM_CLASSES = 1000
# lr incerases linearly with batch size
args.learning_rate = args.learning_rate * (args.batch_size / 256)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device %s' % args.gpu)
    logging.info('small regime or regular regime %s' % args.regime)
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

    writer = SummaryWriter(log_dir=args.save)

    # CrossEntropyLabelSmooth for train, CrossEntropyLoss for val
    criterion_smooth = utils.CrossEntropyLabelSmooth(NUM_CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = RandomlyWiredNN(args.base_channels, NUM_CLASSES, args.output_channels, args.regime, graph_info_dicts)
    x = torch.randn(2, 3, 224, 224)
    writer.add_graph(model, (x,))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("FLOPs = %fMB", utils.model_param_flops_in_MB(model, input_res=[224, 224], multiply_adds=False))

    if args.parallel:
        model = nn.DataParallel(model).cuda()
        print("multi GPUs")
    else:
        model = model.cuda()
        print("single GPU")

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    train_queue, valid_queue = preprocess_imagenet(args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    best_acc_top1 = 0.0
    best_acc_top5 = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc_top1, train_acc_top5, train_obj, train_speed = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('train_acc %f', train_acc_top1)
        logging.info('train_speed_per_image %f', train_speed)
        writer.add_scalar('train_loss', train_obj, epoch)

        valid_acc_top1, valid_acc_top5, valid_obj, valid_speed = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc_top1)
        logging.info('valid_speed_per_image %f', valid_speed)
        writer.add_scalar('val_loss', valid_obj, epoch)

        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            utils.save_parm(model, os.path.join(args.save, 'model_top1.pt'))
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
            utils.save_parm(model, os.path.join(args.save, 'model_top5.pt'))
    writer.close()


def train(train_queue, model, criterion_smooth, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    speed = utils.AvgrageMeter()

    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        n = inputs.size(0)
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        tic = time.time()
        logits = model(inputs)
        toc = time.time()
        loss = criterion_smooth(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        per_image_speed = 1.0 * (toc - tic) / n
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        speed.update(per_image_speed, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f %f', step, objs.avg, top1.avg, top5.avg, speed.avg)

    return top1.avg, top5.avg, objs.avg, speed.avg


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

    return top1.avg, top5.avg, objs.avg, speed.avg


if __name__ == "__main__":
    main()
