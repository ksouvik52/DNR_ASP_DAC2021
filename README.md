
## DNR: A Tunable Robust Pruning FrameworkThrough Dynamic Network Rewiring of DNNs

<p align="center"><img width="30%" src="/images/ASP_DAC_logo.png"></p><br/> 

This repo contains the test codes and saved models of our ASP-DAC 2021 paper: [DNR: A Tunable Robust Pruning FrameworkThrough Dynamic Network Rewiring of DNNs](https://dl.acm.org/doi/10.1145/3394885.3431542)

### Authors:
1. **Souvik Kundu** (souvikku@usc.edu)
2. Mahdi Nazemi (nazemi@usc.edu)
3. Peter A. Beerel (pabeerel@usc.edu)
4. Massoud Pedram (pedram@usc.edu)

#### Robust RESNET saved modes:
1. [ResNet18 Channel pruned on CIFAR-10](https://drive.google.com/file/d/10XndY4udQ6q9eBvnzDtT3NiT2vByItlu/view?usp=sharing)
2. [ResNet18 Irregular pruned on CIFAR-10](https://drive.google.com/file/d/1iqMLZveuFXSgrQa-JtlfOui2ElX1936w/view?usp=sharing)
#### Robust VGG saved modes:
1. [VGG16 Channel pruned on CIFAR-10](https://drive.google.com/file/d/108kCTOxpkDB7aJgsH3ZZvN5EG7y701Si/view?usp=sharing)
2. [VGG16 Irregular pruned on CIFAR-10](https://drive.google.com/file/d/1G4wzZNXL3i7LxGKyjhanuMNvclHrFCYs/view?usp=sharing)

#### To test adversarial accuracy of a saved model, please:
1. Copy and bring the model to same location as the *.py files.
2. Open the run_test.py file to change model_type ['resnet18' or 'vgg16'] and provide the dataset='cifar10'.
3. Provide --adv_eval to enable adversarial evaluation.
4. Run python file: 'python run_test.py'

### Arxiv pre-print version: 
[arxiv_version](https://arxiv.org/abs/2011.03083)

### Cite this work:
      @inproceedings{kundu2021dnr,
      title={DNR: A Tunable Robust Pruning Framework Through Dynamic Network Rewiring of DNNs},
      author={Kundu, Souvik and Nazemi, Mahdi and Beerel, Peter A and Pedram, Massoud},
      booktitle={Proceedings of the 26th Asia and South Pacific Design Automation Conference},
      pages={344--350},
      year={2021}
      }
