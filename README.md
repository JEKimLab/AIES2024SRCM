# Representation Magnitude has a Liability to Privacy Vulnerability
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)](https://pytorch.org/)
[![SciKit-Learn](https://img.shields.io/badge/scikit--learn-1.2.2-yellow?style=flat-square)](https://scikit-learn.org/stable/index.html)

The official Implementation of ***AIES25 - Representation Magnitude has a Liability to Privacy Vulnerability***.

## Tips
1. Check the setting and conf files before using this repo

|      Lib      | Version |
|:-------------:|:-------:|
|    PyYAML     |   6.0   |
| scikit-learn  |  1.2.2  |
| tensorboardX  |   2.6   |
|    pandas     |  2.0.1  |
| matplotlib    |  3.7.1  |


## Running Experiments
You can use the cmd below to train a target model, shadow models and attack model:
```main
python 'main.py' train $arch 
  --arch-conf $arch_conf \
  --dataset $dataset --epoch-file $epoch_file \
  --batch-size $batch_size \
  --learn-method $learn_method \
  --loss-conf $loss_conf \
  --attack-method $attack_method --attack-arch $attack_arch --k $k \
  --if-restore $if_restore --train-target $train_target --train-shadow $train_shadow \
  --save-folder $path
```

An example is shown below:
```
python 'main.py' train resnet 
  --arch-conf resnet18 \
  --dataset cifar100 --epoch-file cifar100_resnet \
  --batch-size 256 \
  --learn-method vanilla \
  --loss-conf vanilla \
  --attack-method vanilla --attack-arch linear --k 5 \
  --if-restore no \
  --save-folder "save_checkpoints/$loss_conf/$dataset/$arch_conf/${learn_method}_${attack_method}/$run/"
```

Datasets include CIFAR-10, CIFAR-100, and Purchase100.

When testing, you can write your own evaluation codes or use `summary_acc_parameter.py`

## Acknowledge
Some portion of codes are derived from:
- [CRL](https://github.com/JEKimLab/UAI24_CRL)
## Other References
- [RelaxLoss](https://github.com/DingfanChen/RelaxLoss)
- [Distillation for Membership Privacy (DMP)](https://github.com/vrt1shjwlkr/AAAI21-MIA-Defense/blob/master/purchase/purchase_distillation.py#L60)

## citation
```
@inproceedings{
}
```
