#! /bin/sh
set -x
export PYTHONPATH=$PYTHONPATH:$PWD

# TITAN
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/wresnet40x2_cifar10_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/wresnet28x10_cifar10_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# TITAN
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/shake26_2x32d_cifar_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/shake26_2x96d_cifar_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0
SEED=0
DATASET=cifar10
CONF=confs/shake26_2x112d_cifar_b512_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0,1,2,3,4,5,6,7
SEED=0
DATASET=cifar10
CONF=confs/pyramid272_cifar10_b64_test.yaml
GENOTYPE=CIFAR10
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# TITAN
GPUS=0
SEED=0
DATASET=cifar100
CONF=confs/wresnet40x2_cifar100_b512_test.yaml
GENOTYPE=CIFAR100
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}


# V100
GPUS=0
SEED=0
DATASET=cifar100
CONF=confs/wresnet28x10_cifar100_b512_test.yaml
GENOTYPE=CIFAR100
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0
SEED=0
DATASET=cifar100
CONF=confs/shake26_2x96d_cifar100_b512_test.yaml
GENOTYPE=CIFAR100
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0,1,2,3,4,5,6,7
SEED=0
DATASET=cifar100
CONF=confs/pyramid272_cifar100_b64_test.yaml
GENOTYPE=CIFAR100
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# TITAN
GPUS=0,1,2,3,4,5,6,7
SEED=0
DATASET=svhn
CONF=confs/wresnet28x10_svhn_b1024_test.yaml
GENOTYPE=SVHN
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# TITAN
GPUS=0,1,2,3,4,5,6,7
SEED=0
DATASET=svhn
CONF=confs/shake26_2x96d_svhn_b1024_test.yaml
GENOTYPE=SVHN
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# TITAN
GPUS=0,1,2,3,4,5,6,7
SEED=0
DATASET=svhn
CONF=confs/shake26_2x96d_svhn_b1024_test.yaml
GENOTYPE=SVHN
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}

# V100
GPUS=0,1,2,3,4,5,6,7
SEED=1
DATASET=imagenet
# CONF=confs/resnet50_b512_test.yaml
CONF=confs/resnet50_b1024_test.yaml
GENOTYPE=ImageNet
SAVE=weights/`basename ${CONF} .yaml`_${GENOTYPE}_${DATASET}_${SEED}/test.pth
CUDA_VISIBLE_DEVICES=${GPUS} python FastAutoAugment/train.py -c ${CONF} --dataset ${DATASET} --genotype ${GENOTYPE} --save ${SAVE} --seed ${SEED}
