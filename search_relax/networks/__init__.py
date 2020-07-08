from __future__ import print_function
# from __future__ import absolute_import
import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
# from torchvision import models

from .resnet import ResNet
from .pyramidnet import PyramidNet
from .shakeshake.shake_resnet import ShakeResNet
from .wideresnet import WideResNet
from .shakeshake.shake_resnext import ShakeResNeXt


def get_model(model_name='wresnet40_2', num_class=10, use_cuda=True, data_parallel=False):
    name = model_name

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)

    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)

    elif name == 'pyramid':
        # model = PyramidNet('cifar10', depth=conf['depth'], alpha=conf['alpha'], num_classes=num_class, bottleneck=conf['bottleneck'])
        raise NotImplementedError
    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        if use_cuda:
            model = model.cuda()
        # import horovod.torch as hvd
        # device = torch.device('cuda', hvd.local_rank())
        # model = model.to(device)
    # cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
