
## DNR: A Tunable Robust Pruning FrameworkThrough Dynamic Network Rewiring of DNNs
<p align="center"><img width="30%" src="/images/ASP_DAC_logo.png"></p><br/> 


#### Robust RESNET saved modes:
1. [ResNet18 Channel pruned on CIFAR-10](https://drive.google.com/file/d/1kbyl34OTxt7YBJON6VpdMBNQWejWpGZN/view?usp=sharing)
2. [ResNet18 Irregular pruned on CIFAR-10](https://drive.google.com/file/d/1nmAPjhM0Hlo2I7k6UPeGDta8g2kN6xJI/view?usp=sharing)
#### Robust VGG saved modes:
1. [VGG16 Channel pruned on CIFAR-10](https://drive.google.com/file/d/1hzeTRoFo0vaPVqRhVQjr80ugsSk6zlvh/view?usp=sharing)
2. [VGG16 Irregular pruned on CIFAR-10](https://drive.google.com/file/d/1-aYjjBaulln_nfxagF6fhsYuToG-FSBk/view?usp=sharing)

#### To test adversarial accuracy of a saved model, please:
1. Copy and bring the model to same location as the *.py files.
2. Open the run_test.py file to change model_type ['resnet18' or 'vgg16'] and provide the dataset='cifar10'.
3. Provide --adv_eval to enable adversarial evaluation.
4. Run python file: 'python run_test.py'

### Cite this work:
      @misc
      {8919683, 
      author    ={S. {Kundu} and S. {Prakash} and H. {Akrami} and P. A. {Beerel} and K. M. {Chugg}}, 
      booktitle ={2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton)}, 
      title     ={pSConv: A Pre-defined Sparse Kernel Based Convolution for Deep CNNs}, 
      year      ={2019}, 
      pages     ={100-107},}
  @misc
  {kundu2020tunable,
  title={A Tunable Robust Pruning Framework Through Dynamic Network Rewiring of DNNs}, 
  author={Souvik Kundu and Mahdi Nazemi and Peter A. Beerel and Massoud Pedram},
  year={2020},
  journal={arXiv preprint arXiv:2011.03083},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
 }
