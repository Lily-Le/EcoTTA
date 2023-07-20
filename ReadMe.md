This is a (unaccomplished) community implementation for the paper [EcoTTA](https://arxiv.org/abs/2303.01904).  

Reference code:  
[pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)   
[EATA](https://github.com/mr-eggplant/EATA/blob/main)  
[TENT](https://github.com/DequanWang/tent)  
 Folder Robustbench cloned from [Robustbench](https://github.com/RobustBench/robustbench)  
 Folder autoattack cloned from [Auto-attack](https://github.com/fra31/auto-attack)  

Dataset: CIFAR10, CIFAR-100  
Model: WideResNet28, WideResNet40

# Usage
For experiment on CIFAR-100 and WideResNet40 
## warmup phase:  
>run   
`python main.py --dataset cifar100 --net wideresnet40 --b 64 --mode pretrain --warm 1 --lr 5e-2 --warmup_epoch 10 --checkpoint_path [path to save checkpoints] --log_dir_pretrain [path to save tensorboard logs]` 


The checkpoint and txt log file will be saved in `./checkpoint_path/*`. The tensorboard log will be saved in `./log_dir_pretrain/*`. 

The required robust model will be automatically downloaded at `robust_models` folder.

## adaptation phase:  
The corruption config is in `conf.py`. By default you don't have to change it.  
To experiment on specific corruptions or severity, you can change the configures in `conf.py` accordingly.  
>run  
`python main.py --dataset cifar100 --net wideresnet40 --b 64 --mode tta --lr_tta 5e-3 --warmup_checkpoint [your warmup checkpoint path] --e_margin 0.4 --lambda_reg 0.25 --log_dir_tta [log directory for tensorboard logs]`  

The txt log for adaptation phase will be saved at the same folder of your warmup checkpoint.

# Reproduction result
CIFAR-100 WideResNet40  
mean err = 0.36481.

Other results to be updated.
# Citation:
```bibtex
@inproceedings{song2023ecotta,
  title={EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled Regularization},
  author={Junha Song and Jungsoo Lee and In So Kweon and Sungha Choi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

Thank the authors for their marvelous job!   
I hope that I've faithfully reproduced their work. Other results (ResNet50 ImageNet, combination with adaptBN, etc.) will be updated when I'm not so busy. If there is any problem, welcome to raise issues.