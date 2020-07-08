set -x
DATASET=reduced_imagenet
DATASET=reduced_cifar10
DATASET=reduced_cifar100
DATASET=reduced_svhn
MODEL=resnet50
MODEL=wresnet40_2
MODEL=wresnet28_10
GPU=3
EPOCH=100
EPOCH=20
BATCH=64
BATCH=128
LR=0.0025
LR=0.0125
LR=0.1
LR=0.005
WD=0.001
AWD=0.0
ALR=0.01

# svhn
DATASET=reduced_svhn
MODEL=wresnet28_10
EPOCH=100
BATCH=128
LR=0.005
WD=0.001
AWD=0.0
ALR=0.01

# cifar10
GPU=3
DATASET=reduced_cifar10
MODEL=wresnet40_2
EPOCH=20
BATCH=32
LR=0.025
WD=0.0002
AWD=0.0
ALR=0.001
CUTOUT=16

# cifar100
# GPU=3
# DATASET=reduced_cifar100
# MODEL=wresnet40_2
# EPOCH=20
# BATCH=32
# LR=0.025
# WD=0.0005
# AWD=0.0
# ALR=0.001

# cifar10
GPU=3
DATASET=cifar10
MODEL=wresnet40_2
EPOCH=200
BATCH=512
LR=0.4
WD=0.0002
AWD=0.0
ALR=0.001
CUTOUT=16

#202002171835
GPU=2
DATASET=reduced_cifar10
MODEL=wresnet40_2
EPOCH=20
BATCH=128
LR=0.1
WD=0.0002
AWD=0.0
ALR=0.005
CUTOUT=16
TEMPERATURE=0.5





SAVE=paper_augment2_stp_multi_${DATASET}_${MODEL}_${BATCH}_${EPOCH}_awd${AWD}_alr${ALR}_cutout_${CUTOUT}_lr${LR}_wd${WD}_temp_${TEMPERATURE}
which python
python train_search_paper.py --unrolled --report_freq 1 --num_workers 0 --epoch ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --arch_weight_decay ${AWD} --arch_learning_rate ${ALR} --weight_decay ${WD} --cutout --cutout_length ${CUTOUT} --temperature ${TEMPERATURE}
