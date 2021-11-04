# `DADA: Differentiable Automatic Data Augmentation`
Contact us with liyonggang@pku.edu.cn, wyt@pku.edu.cn.

## Introduction
The official code for our ECCV 2020 paper `DADA: Differentiable Automatic Data Augmentation`, which is at least one order of magnitude faster than
the state-of-the-art data augmentation (DA) policy search algorithms while achieving very comparable accuracy.
The implementation of our training part is based on [fast-autoaugment](https://github.com/kakaobrain/fast-autoaugment).

## License
**The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.**

## Citation
If you use our code/model, please consider to cite our ECCV 2020 paper **DADA: Differentiable Automatic Data Augmentation** [[arXiv](https://arxiv.org/pdf/2003.03780.pdf)] [[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_35)].

```bibtex
@article{li2020dada,
  author    = {Yonggang Li and
               Guosheng Hu and
               Yongtao Wang and
               Timothy M. Hospedales and
               Neil Martin Robertson and
               Yongxin Yang},
  title     = {{DADA:} Differentiable Automatic Data Augmentation},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

## Model
We provide the checkpoints in [BaiduDrive](https://pan.baidu.com/s/17VVe_U9BwzBoE4pI5eA_vQ), with fetching code **sgap**, or [GoogleDrive](https://drive.google.com/file/d/13LXk-Nw-g7RZ6gP6oMNEByOw4OVBdqyS/view?usp=sharing).

### CIFAR-10
Search : **0.1 GPU Hours**, WResNet-40x2 on Reduced CIFAR-10


Dataset  | Model | Baseline | Cutout | AA | PBA | Fast AA | DADA 
---------|------------------|-------|-------|------|--------|-------|---
CIFAR-10 | Wide-ResNet-40-2 | 5.3   | 4.1   | 3.7   | -     | 3.6   | 3.6
CIFAR-10 | Wide-ResNet-28-10 | 3.9   | 3.1   | 2.6   | 2.6   | 2.7   | 2.7
CIFAR-10 | Shake-Shake(26 2x32d) | 3.6   | 3.0     | 2.5   | 2.5   | 2.7   | 2.7
CIFAR-10 | Shake-Shake(26 2x96d) | 2.9   | 2.6   | 2.0     | 2.0     | 2.0     | 2.0
CIFAR-10 | Shake-Shake(26 2x112d) | 2.8   | 2.6   | 1.9   | 2.0     | 2.0     | 2.0
CIFAR-10 | PyramidNet+ShakeDrop | 2.7   | 2.3   | 1.5   | 1.5   | 1.8   | 1.7 

### CIFAR-100
Search : **0.2 GPU Hours**, WResNet-40x2 on Reduced CIFAR-100


Dataset| Model | Baseline | Cutout | AA | PBA | Fast AA | DADA 
---------|------------------|-------|-------|------|--------|-------|---
CIFAR-100 | Wide-ResNet-40-2 | 26.0    | 25.2  | 20.7  | -     | 20.7  | 20.9
CIFAR-100 | Wide-ResNet-28-10 | 18.8  | 18.4  | 17.1  | 16.7  | 17.3  | 17.5 
CIFAR-100 | Shake-Shake(26 2x96d) | 17.1  | 16.0    | 14.3  | 15.3  | 14.9  | 15.3 
CIFAR-100 | PyramidNet+ShakeDrop | 14.0    | 12.2  | 10.7  | 10.9  | 11.9  | 11.2


### SVHN
Search : **0.1 GPU Hours**, WResNet-28x10 on Reduced SVHN


Dataset| Model | Baseline | Cutout | AA | PBA | Fast AA | DADA 
---------|------------------|-------|-------|------|--------|-------|---
SVHN | Wide-ResNet-28-10 | 1.5   | 1.3   | 1.1   | 1.2   | 1.1   | 1.2 
SVHN | Shake-Shake(26 2x96d) | 1.4   | 1.2   | 1.0  | 1.1  | -     | 1.1 


### ImageNet
Search : **1.3 GPU Hours**, ResNet-50 on Reduced ImageNet


Dataset| Baseline | AA | Fast AA | OHL AA | DADA 
---------|------------------|-------|-------|------|--------
ImageNet | 23.7 / 6.9 | ~22.4 / 6.2 | 22.4 / 6.3 | 21.1 / 5.7 | 22.5 / 6.5 



## Installation

### Environment

1. Ubuntu 16.04 LTS
2. CUDA 10.0
3. PyTorch 1.2.0
4. TorchVision 0.4.0

### Install
a. Create a conda virtual environment and activate it.

```shell
conda create -n dada-env python=3.6.10
source activate dada-env # or conda activate dada-env
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit==10.0
```

c. Install other python package for DADA and fast-autoaugment, e.g.,

```shell
# for training and inference
pip install -r fast-autoaugment/requirements.txt

# for searching
pip install -r requirements.txt
```

## Getting Started

### Prepare Datasets
The dataset (except ImageNet) will be automatically download if you keep the default setting.
You should put the data in ./data as below: (which include the datasets of CIFAR-10, CIFAR-100, SVHN, and ImageNet)
```shell
# CIFAR-10
./data/cifar-10-python.tar.gz

# CIFAR-100
./data/cifar-100-python.tar.gz

# SVHN
./data/train_32x32.mat
./data/extra_32x32.mat
./data/test_32x32.mat

# ImageNet
./data/imagenet-pytorch/
./data/imagenet-pytorch/meta.bin
./data/imagenet-pytorch/train
./data/imagenet-pytorch/val
```

### Inference
Download the model-pth provided in , put them in `./fast-autoaugment/weights`
```shell
cd fast-autoaugment
sh inference.sh
```

For example, you can test the provided wresnet40x2 model trained on CIFAR-10 as below:
```shell
# TITAN
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/wresnet40x2_cifar10_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED} --only-eval --batch 32
```

### Train
The training script is provided, including most experiments of our paper.
```shell
cd fast-autoaugment
sh train.sh
```

For example, you can train a wresnet40x2 model on CIFAR-10 as below:
```shell
# TITAN
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/wresnet40x2_cifar10_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}
```

### Search
The searching script is provided, including CIFAR10, CIFAR100, SVHN, and ImageNet.
```shell
cd search_relax
sh train_paper.sh
```

For example, you can search a DA policy on the reduced-cifar10 dataset with wresnet40-2 model as below:
```shell
# you can change the hyper-parameters as below:
GPU=0
DATASET=reduced_cifar10
MODEL=wresnet40_2
EPOCH=20
BATCH=128
LR=0.1
WD=0.0002
AWD=0.0
ALR=0.005
CUTOUT=16
TEMPERATE=0.5
SAVE=CIFAR10
python train_search_paper.py --unrolled --report_freq 1 --num_workers 0 --epoch ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --arch_weight_decay ${AWD} --arch_learning_rate ${ALR} --weight_decay ${WD} --cutout --cutout_length ${CUTOUT} --temperature ${TEMPERATE}
```

The code for DADA with gumbel softmax is also included in this repository.
```shell
cd search_gumbel
sh train_paper.sh
```





## Found Policy
We relase the found Data Augmentation policies in CIFAR-10, CIFAR-100, SVHN, and ImageNet by our DADA as below. 
The origin DA policies have been included in the `fast-autoaugment/FastAutoAugment/genotype.py`.
You can find the genotype used by our paper as below:
```shell
vim fast-autoaugment/FastAutoAugment/genotype.py
```
### CIFAR10

Sub-policy | Opeartion 1 | Opeartion 2|
---|---|---
sub-policy 0 | (TranslateX, 0.52, 0.58) | (Rotate, 0.57, 0.53)
sub-policy 1 | (ShearX, 0.50, 0.46) | (Sharpness, 0.50, 0.54)
sub-policy 2 | (Brightness, 0.56, 0.56) | (Sharpness, 0.52, 0.47)
sub-policy 3 | (ShearY, 0.62, 0.48) | (Brightness, 0.47, 0.46)
sub-policy 4 | (ShearX, 0.44, 0.58) | (TranslateY, 0.40, 0.51)
sub-policy 5 | (Rotate, 0.40, 0.52) | (Equalize, 0.38, 0.36)
sub-policy 6 | (AutoContrast, 0.44, 0.48) | (Cutout, 0.49, 0.50)
sub-policy 7 | (AutoContrast, 0.56, 0.48) | (Color, 0.45, 0.61)
sub-policy 8 | (Rotate, 0.42, 0.64) | (AutoContrast, 0.60, 0.58)
sub-policy 9 | (Invert, 0.40, 0.50) | (Color, 0.50, 0.44)
sub-policy 10 | (Posterize, 0.56, 0.50) | (Brightness, 0.53, 0.48)
sub-policy 11 | (TranslateY, 0.42, 0.51) | (AutoContrast, 0.38, 0.57)
sub-policy 12 | (ShearX, 0.38, 0.50) | (Contrast, 0.49, 0.52)
sub-policy 13 | (ShearY, 0.54, 0.60) | (Rotate, 0.31, 0.56)
sub-policy 14 | (Posterize, 0.42, 0.50) | (Color, 0.45, 0.56)
sub-policy 15 | (TranslateX, 0.41, 0.45) | (TranslateY, 0.36, 0.48)
sub-policy 16 | (TranslateX, 0.57, 0.50) | (Brightness, 0.54, 0.48)
sub-policy 17 | (TranslateX, 0.53, 0.51) | (Cutout, 0.69, 0.49)
sub-policy 18 | (ShearX, 0.46, 0.44) | (Invert, 0.42, 0.40)
sub-policy 19 | (Rotate, 0.50, 0.42) | (Contrast, 0.49, 0.42)
sub-policy 20 | (Rotate, 0.43, 0.47) | (Solarize, 0.50, 0.42)
sub-policy 21 | (TranslateY, 0.74, 0.51) | (Color, 0.39, 0.57)
sub-policy 22 | (Equalize, 0.42, 0.53) | (Sharpness, 0.40, 0.43)
sub-policy 23 | (Solarize, 0.73, 0.42) | (Cutout, 0.51, 0.46)
sub-policy 24 | (ShearX, 0.58, 0.56) | (TranslateX, 0.48, 0.49)

### CIFAR-100
Sub-policy | Opeartion 1 | Opeartion 2|
---|---|---
sub-policy 0 | (ShearY, 0.56, 0.28) | (Sharpness, 0.49, 0.22)
sub-policy 1 | (Rotate, 0.36, 0.19) | (Contrast, 0.56, 0.31)
sub-policy 2 | (TranslateY, 0.00, 0.41) | (Brightness, 0.47, 0.52)
sub-policy 3 | (AutoContrast, 0.80, 0.44) | (Color, 0.44, 0.37)
sub-policy 4 | (Color, 0.94, 0.25) | (Brightness, 0.68, 0.45)
sub-policy 5 | (TranslateY, 0.63, 0.40) | (Equalize, 0.82, 0.30)
sub-policy 6 | (Equalize, 0.46, 0.71) | (Posterize, 0.50, 0.72)
sub-policy 7 | (Color, 0.52, 0.48) | (Sharpness, 0.19, 0.40)
sub-policy 8 | (Sharpness, 0.42, 0.38) | (Cutout, 0.55, 0.24)
sub-policy 9 | (ShearX, 0.74, 0.56) | (TranslateX, 0.48, 0.67)
sub-policy 10 | (Invert, 0.36, 0.59) | (Brightness, 0.50, 0.23)
sub-policy 11 | (TranslateX, 0.36, 0.36) | (Posterize, 0.80, 0.32)
sub-policy 12 | (TranslateX, 0.48, 0.36) | (Cutout, 0.64, 0.67)
sub-policy 13 | (Posterize, 0.31, 0.04) | (Contrast, 1.00, 0.08)
sub-policy 14 | (Contrast, 0.42, 0.26) | (Cutout, 0.00, 0.44)
sub-policy 15 | (Equalize, 0.16, 0.69) | (Brightness, 0.73, 0.18)
sub-policy 16 | (Contrast, 0.45, 0.34) | (Sharpness, 0.59, 0.28)
sub-policy 17 | (TranslateX, 0.13, 0.54) | (Invert, 0.33, 0.48)
sub-policy 18 | (Rotate, 0.50, 0.58) | (Posterize, 1.00, 0.74)
sub-policy 19 | (TranslateX, 0.51, 0.43) | (Rotate, 0.46, 0.48)
sub-policy 20 | (ShearX, 0.58, 0.46) | (TranslateY, 0.33, 0.31)
sub-policy 21 | (Rotate, 1.00, 0.00) | (Equalize, 0.51, 0.37)
sub-policy 22 | (AutoContrast, 0.26, 0.57) | (Cutout, 0.34, 0.35)
sub-policy 23 | (ShearX, 0.56, 0.55) | (Color, 0.50, 0.50)
sub-policy 24 | (ShearY, 0.46, 0.09) | (Posterize, 0.55, 0.34)

### SVHN
Sub-policy | Opeartion 1 | Opeartion 2|
---|---|---
sub-policy 0 | (Solarize, 0.61, 0.53) | (Brightness, 0.64, 0.50)
sub-policy 1 | (ShearY, 0.56, 0.54) | (Sharpness, 0.67, 0.50)
sub-policy 2 | (AutoContrast, 0.64, 0.50) | (Posterize, 0.49, 0.42)
sub-policy 3 | (Invert, 0.43, 0.62) | (Equalize, 0.30, 0.53)
sub-policy 4 | (Contrast, 0.49, 0.55) | (Color, 0.51, 0.58)
sub-policy 5 | (ShearX, 0.58, 0.50) | (Brightness, 0.56, 0.54)
sub-policy 6 | (Rotate, 0.43, 0.50) | (Contrast, 0.47, 0.42)
sub-policy 7 | (Brightness, 0.51, 0.57) | (Cutout, 0.48, 0.50)
sub-policy 8 | (TranslateY, 0.65, 0.46) | (Rotate, 0.43, 0.46)
sub-policy 9 | (ShearY, 0.41, 0.43) | (Contrast, 0.48, 0.49)
sub-policy 10 | (ShearY, 0.52, 0.37) | (Brightness, 0.43, 0.37)
sub-policy 11 | (ShearY, 0.26, 0.49) | (Posterize, 0.52, 0.56)
sub-policy 12 | (TranslateX, 0.67, 0.38) | (TranslateY, 0.45, 0.42)
sub-policy 13 | (Posterize, 0.64, 0.43) | (Sharpness, 0.63, 0.54)
sub-policy 14 | (Rotate, 0.47, 0.50) | (Sharpness, 0.40, 0.45)
sub-policy 15 | (ShearX, 0.47, 0.46) | (Cutout, 0.58, 0.50)
sub-policy 16 | (Rotate, 0.58, 0.53) | (Solarize, 0.41, 0.43)
sub-policy 17 | (Color, 0.37, 0.44) | (Brightness, 0.52, 0.41)
sub-policy 18 | (TranslateX, 0.49, 0.47) | (Posterize, 0.49, 0.52)
sub-policy 19 | (TranslateY, 0.50, 0.49) | (Solarize, 0.50, 0.42)
sub-policy 20 | (TranslateY, 0.27, 0.50) | (Invert, 0.56, 0.29)
sub-policy 21 | (ShearY, 0.64, 0.57) | (Rotate, 0.49, 0.57)
sub-policy 22 | (Invert, 0.49, 0.55) | (Contrast, 0.41, 0.50)
sub-policy 23 | (ShearX, 0.57, 0.49) | (Color, 0.60, 0.50)
sub-policy 24 | (Rotate, 0.54, 0.53) | (Equalize, 0.52, 0.50)

### ImageNet
Sub-policy | Opeartion 1 | Opeartion 2|
---|---|---
sub-policy 0 | (TranslateY, 0.85, 0.64) | (Contrast, 0.70, 0.47)
sub-policy 1 | (ShearX, 0.69, 0.64) | (Brightness, 0.58, 0.46)
sub-policy 2 | (Solarize, 0.33, 0.53) | (Contrast, 0.36, 0.40)
sub-policy 3 | (ShearY, 0.54, 0.81) | (Color, 0.65, 0.67)
sub-policy 4 | (Rotate, 0.52, 0.28) | (Invert, 0.55, 0.46)
sub-policy 5 | (ShearY, 0.76, 0.55) | (AutoContrast, 0.64, 0.46)
sub-policy 6 | (TranslateX, 0.32, 0.67) | (Sharpness, 0.45, 0.61)
sub-policy 7 | (Brightness, 0.28, 0.54) | (Cutout, 0.29, 0.53)
sub-policy 8 | (TranslateY, 0.26, 0.39) | (Brightness, 0.30, 0.57)
sub-policy 9 | (ShearX, 0.46, 0.62) | (Rotate, 0.51, 0.59)
sub-policy 10 | (TranslateY, 0.63, 0.38) | (Invert, 0.40, 0.33)
sub-policy 11 | (TranslateY, 0.49, 0.32) | (Equalize, 0.43, 0.26)
sub-policy 12 | (TranslateX, 0.31, 0.46) | (AutoContrast, 0.40, 0.00)
sub-policy 13 | (ShearY, 0.57, 0.35) | (Equalize, 0.45, 0.16)
sub-policy 14 | (Solarize, 0.78, 0.61) | (Brightness, 0.57, 0.80)
sub-policy 15 | (Color, 0.75, 0.40) | (Cutout, 0.54, 0.47)
sub-policy 16 | (ShearX, 0.51, 0.67) | (Cutout, 0.37, 0.45)
sub-policy 17 | (TranslateX, 0.68, 0.39) | (Rotate, 0.47, 0.16)
sub-policy 18 | (Rotate, 0.64, 0.55) | (Sharpness, 0.66, 0.80)
sub-policy 19 | (TranslateY, 0.47, 0.75) | (Sharpness, 0.64, 0.52)
sub-policy 20 | (AutoContrast, 0.29, 0.54) | (Posterize, 0.35, 0.70)
sub-policy 21 | (Invert, 0.55, 0.49) | (Equalize, 0.44, 0.76)
sub-policy 22 | (TranslateX, 0.86, 0.29) | (Contrast, 0.41, 0.60)
sub-policy 23 | (Invert, 0.28, 0.45) | (Posterize, 0.42, 0.34)
sub-policy 24 | (Posterize, 0.15, 0.33) | (Color, 0.50, 0.59)


