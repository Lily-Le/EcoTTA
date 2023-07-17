# Readme
This is a (unaccomplished) community implementation for the paper [EcoTTA](https://arxiv.org/abs/2303.01904).  

Reference code:  
[pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)   
[EATA](https://github.com/mr-eggplant/EATA/blob/main)  
[TENT](https://github.com/DequanWang/tent)  
 Folder Robustbench cloned from [Robustbench](https://github.com/RobustBench/robustbench)  
 Folder autoattack cloned from [Auto-attack](https://github.com/fra31/auto-attack)  

Dataset: CIFAR10, CIFAR-100  
Model: WideResNet28, WideResNet40

run `main.py`. Change the argumernts for warmup / adaptation process.

Reproduction result for cifar100 wideresnet40: mean err = 0.36481.

Other results to be updated.
### Citation:
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