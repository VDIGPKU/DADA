set -x
# cifar10
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
SAVE=paper_augment2_st_multi_relax_fix_${DATASET}_${MODEL}_${BATCH}_${EPOCH}_awd${AWD}_alr${ALR}_cutout_${CUTOUT}_lr${LR}_wd${WD}_temp_${TEMPERATE}_seed_${SEED}
SAVE=CIFAR10
python train_search_paper.py --unrolled --report_freq 1 --num_workers 0 --epoch ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --arch_weight_decay ${AWD} --arch_learning_rate ${ALR} --weight_decay ${WD} --cutout --cutout_length ${CUTOUT} --temperature ${TEMPERATE}

# cifar100
GPU=0
DATASET=reduced_cifar100
MODEL=wresnet40_2
EPOCH=20
BATCH=32
LR=0.1
WD=0.0005
AWD=0.0
ALR=0.005
CUTOUT=16
TEMPERATE=0.5
SAVE=paper_augment2_st_multi_relax_fix_${DATASET}_${MODEL}_${BATCH}_${EPOCH}_awd${AWD}_alr${ALR}_cutout_${CUTOUT}_lr${LR}_wd${WD}_temp_${TEMPERATE}_seed_${SEED}
SAVE=CIFAR100
python train_search_paper.py --unrolled --report_freq 1 --num_workers 0 --epoch ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --arch_weight_decay ${AWD} --arch_learning_rate ${ALR} --weight_decay ${WD} --cutout --cutout_length ${CUTOUT} --temperature ${TEMPERATE}

# svhn
GPU=0
DATASET=reduced_svhn
MODEL=wresnet28_10
EPOCH=20
BATCH=32
LR=0.00125
WD=0.001
AWD=0.0
ALR=0.005
CUTOUT=16
TEMPERATE=0.5
SAVE=paper_augment2_st_multi_relax_fix_${DATASET}_${MODEL}_${BATCH}_${EPOCH}_awd${AWD}_alr${ALR}_cutout_${CUTOUT}_lr${LR}_wd${WD}_temp_${TEMPERATE}_seed_${SEED}
SAVE=SVHN
python train_search_paper.py --unrolled --report_freq 1 --num_workers 0 --epoch ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --arch_weight_decay ${AWD} --arch_learning_rate ${ALR} --weight_decay ${WD} --cutout --cutout_length ${CUTOUT} --temperature ${TEMPERATE}

# imagenet
GPU=0
DATASET=reduced_imagenet
MODEL=resnet50
EPOCH=20
BATCH=32
LR=0.00125
WD=0.0001
AWD=0.0
ALR=0.005
CUTOUT=16
TEMPERATE=0.5
SAVE=paper_augment2_st_multi_relax_fix_${DATASET}_${MODEL}_${BATCH}_${EPOCH}_awd${AWD}_alr${ALR}_cutout_${CUTOUT}_lr${LR}_wd${WD}_temp_${TEMPERATE}_seed_${SEED}
SAVE=ImageNet
python train_search_paper.py --unrolled --report_freq 1 --num_workers 0 --epoch ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --arch_weight_decay ${AWD} --arch_learning_rate ${ALR} --weight_decay ${WD} --cutout --cutout_length ${CUTOUT} --temperature ${TEMPERATE}
