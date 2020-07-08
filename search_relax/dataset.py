import torch
import torchvision
from operation import apply_augment
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
from primitives import sub_policies
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
from imagenet import ImageNet
from operation import Lighting
import os
import numpy as np

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
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

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pre_transforms, after_transforms, valid_transforms, ops_names, search, magnitudes):
        super(AugmentDataset, self).__init__()
        self.dataset = dataset
        self.pre_transforms = pre_transforms
        self.after_transforms = after_transforms
        self.valid_transforms = valid_transforms
        self.ops_names = ops_names
        self.search = search
        self.magnitudes = magnitudes

    def __getitem__(self, index):
        if self.search:
            # start_time = time.time()
            img, target = self.dataset.__getitem__(index)
            img = self.pre_transforms(img)
            magnitude = self.magnitudes.clamp(0, 1)[self.weights_index.item()]
            sub_policy = self.ops_names[self.weights_index.item()]
            probability_index = self.probabilities_index[self.weights_index.item()]
            image = img
            for i, ops_name in enumerate(sub_policy):
                if probability_index[i].item() != 0.0:
                    image = apply_augment(image, ops_name, magnitude[i])
            image = self.after_transforms(image)
            return image, target

            # outs = [None for i in range(2**len(sub_policy)) ]
            # def dfs(image, index, depth):
            #     if depth == len(sub_policy):
            #         # print(index)
            #         outs[index] = self.after_transforms(image)
            #         return
            #     dfs(image, index, depth+1)
            #     new_image = apply_augment(image, sub_policy[depth], magnitude[depth])
            #     dfs(new_image, (1<<depth) + index, depth+1)
            # dfs(img, 0, 0)
            # image = img
            # for i, ops_name in enumerate(sub_policy):
            #     image = apply_augment(image, ops_name, magnitude[i])
            # image = self.after_transforms(image)
            # print(self.magnitudes)
            # print(self.weights_index)
            # end_time = time.time()
            # print("%f" % (end_time - start_time))
            # return tuple(outs), target
        else:
            img, target = self.dataset.__getitem__(index)
            if self.valid_transforms is not None:
                img = self.valid_transforms(img)
            return img, target

    def __len__(self):
        return self.dataset.__len__()

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]

def get_dataloaders(dataset, batch, num_workers, dataroot, ops_names, magnitudes, cutout, cutout_length, split=0.5, split_idx=0, target_lb=-1):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train_pre = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        transform_train_after = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        transform_train_pre = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
        ])
        transform_train_after = transforms.Compose([
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    if cutout and cutout_length != 0:
        transform_train_after.transforms.append(CutoutDefault(cutout_length))

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
        # testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)
    elif dataset == 'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        # testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=None)
        # testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=None)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        # testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=None)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=None)
        total_trainset = ConcatDataset([trainset, extraset])
        # testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'reduced_svhn':
        total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=None)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-1000, random_state=0)  # 1000 trainset
        # sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.labels)
        train_idx, valid_idx = next(sss)
        # targets = [total_trainset.targets[idx] for idx in train_idx]
        targets = [total_trainset.labels[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        # total_trainset.targets = targets
        total_trainset.labels = targets
        total_trainset.targets = targets

        # testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=None)
        # testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
#         idx120 = sorted(random.sample(list(range(1000)), k=120))
        idx120 = [16, 23, 52, 57, 76, 93, 95, 96, 99, 121, 122, 128, 148, 172, 181, 189, 202, 210, 232, 238, 257, 258, 259, 277, 283, 289, 295, 304, 307, 318, 322, 331, 337, 338, 345, 350, 361, 375, 376, 381, 388, 399, 401, 408, 424, 431, 432, 440, 447, 462, 464, 472, 483, 497, 506, 512, 530, 541, 553, 554, 557, 564, 570, 584, 612, 614, 619, 626, 631, 632, 650, 657, 658, 660, 674, 675, 680, 682, 691, 695, 699, 711, 734, 736, 741, 754, 757, 764, 769, 770, 780, 781, 787, 797, 799, 811, 822, 829, 830, 835, 837, 842, 843, 845, 873, 883, 897, 900, 902, 905, 913, 920, 925, 937, 938, 940, 941, 944, 949, 959]
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=None)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=None)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 50000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)

        # filter out
        # train_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, train_idx))
        # valid_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, valid_idx))
        # test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))
        train_idx = list(filter(lambda x: total_trainset.targets[x] in idx120, train_idx))
        valid_idx = list(filter(lambda x: total_trainset.targets[x] in idx120, valid_idx))
        test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))

        targets = [idx120.index(total_trainset.targets[idx]) for idx in train_idx]
        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        for idx in range(len(testset.samples)):
            if testset.samples[idx][1] not in idx120:
                continue
            testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
        idx120 = [904, 385, 759, 884, 784, 844, 132, 214, 990, 786, 979, 582, 104, 288, 697, 480, 66, 943, 308, 282, 118, 926, 882, 478, 133, 884, 570, 964, 825, 656, 661, 289, 385, 448, 705, 609, 955, 5, 703, 713, 695, 811, 958, 147, 6, 3, 59, 354, 315, 514, 741, 525, 685, 673, 657, 267, 575, 501, 30, 455, 905, 860, 355, 911, 24, 708, 346, 195, 660, 528, 330, 511, 439, 150, 988, 940, 236, 803, 741, 295, 111, 520, 856, 248, 203, 147, 625, 589, 708, 201, 712, 630, 630, 367, 273, 931, 960, 274, 112, 239, 463, 355, 955, 525, 404, 59, 981, 725, 90, 782, 604, 323, 418, 35, 95, 97, 193, 690, 869, 172]
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=None)
        # testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        # sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 6000, random_state=0)  # 4000 trainset
        # sss = StratifiedShuffleSplit(n_splits=1, test_size=0, random_state=0)  # 4000 trainset
        # sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        # train_idx, valid_idx = next(sss)
        # print(len(train_idx), len(valid_idx))

        # filter out
        # train_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, train_idx))
        # valid_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, valid_idx))
        # # test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))
        train_idx = list(range(len(total_trainset)))

        filter_train_idx = list(filter(lambda x: total_trainset.targets[x] in idx120, train_idx))
        # valid_idx = list(filter(lambda x: total_trainset.targets[x] in idx120, valid_idx))
        # test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))
        # print(len(filter_train_idx))

        targets = [idx120.index(total_trainset.targets[idx]) for idx in filter_train_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(filter_train_idx) - 6000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(filter_train_idx))), targets)
        train_idx, valid_idx = next(sss)
        train_idx = [filter_train_idx[x] for x in train_idx]
        valid_idx = [filter_train_idx[x] for x in valid_idx]



        targets = [idx120.index(total_trainset.targets[idx]) for idx in train_idx]
        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        # for idx in range(len(testset.samples)):
        #     if testset.samples[idx][1] not in idx120:
        #         continue
        #     testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        # testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)


        # if horovod:
        #     import horovod.torch as hvd
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        valid_sampler = SubsetSampler([])

        # if horovod:
        #     import horovod.torch as hvd
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(valid_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    train_data = AugmentDataset(total_trainset, transform_train_pre, transform_train_after, transform_test, ops_names, True, magnitudes)
    valid_data = AugmentDataset(total_trainset, transform_train_pre, transform_train_after, transform_test, ops_names, False, magnitudes)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch, shuffle=False,
        sampler=train_sampler, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        sampler=valid_sampler, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    # trainloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=32, pin_memory=True,
    #     sampler=train_sampler, drop_last=True)
    # validloader = torch.utils.data.DataLoader(
    #     total_trainset, batch_size=batch, shuffle=False, num_workers=16, pin_memory=True,
    #     sampler=valid_sampler, drop_last=False)

    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=True,
    #     drop_last=False
    # )
    print(len(train_data))
    return trainloader, validloader
