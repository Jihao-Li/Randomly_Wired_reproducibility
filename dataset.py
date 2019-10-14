import os
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    """
    init cifar10
    :param args: 
    :return: 
    """
    CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]

    # init train set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    # init val set
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, valid_transform


def preprocess_cifar10(args):
    """

    :param args: 
    :return: 
    """
    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    val_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, num_workers=2)

    return train_queue, valid_queue


def _data_transforms_cifar100(args):
    """
    init cifar100
    :param args: 
    :return: 
    """
    CIFAR100_MEAN = [0.5070754, 0.48655024, 0.44091907]
    CIFAR100_STD = [0.26733398, 0.25643876, 0.2761503]

    # init train set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    # init val set
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    return train_transform, valid_transform


def preprocess_cifar100(args):
    """
    
    :param args: 
    :return: 
    """
    train_transform, valid_transform = _data_transforms_cifar100(args)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    val_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, num_workers=2)

    return train_queue, valid_queue


def _data_transforms_imagenet():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # init train set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # init val set
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform


def preprocess_imagenet(args):
    train_transform, valid_transform = _data_transforms_imagenet()

    if args.train_flag:
        train_dir = os.path.join(args.data, 'train')
        val_dir = os.path.join(args.data, 'val')

        train_data = dset.ImageFolder(train_dir, train_transform)
        val_data = dset.ImageFolder(val_dir, valid_transform)

        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=4)
        valid_queue = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=4)

        return train_queue, valid_queue
    else:
        test_dir = os.path.join(args.data, 'test')
        test_data = dset.ImageFolder(test_dir, valid_transform)
        test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=4)

        return None, test_queue


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser("cifar10")
    # parser.add_argument('--data', type=str, default='/dataset/cifar10/', help='location of the data corpus')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    # parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    # args = parser.parse_args()
    #
    # train_queue, valid_queue = preprocess_cifar10(args)
    # for step, (input, target) in enumerate(train_queue):
    #     n = input.size(0)
    #     print(input.size())
    #     print(target.size())

    import argparse
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument('--data', type=str, default='/dataset/extract_ILSVRC2012', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--train_flag', type=bool, default=True, help='train or eval')
    args = parser.parse_args()

    train_queue, valid_queue = preprocess_imagenet(args)
    for step, (inputs, targets) in enumerate(train_queue):
        n = inputs.size(0)
        print(inputs.size())
        print(targets.size())
