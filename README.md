
## DNR: A Tunable Robust Pruning FrameworkThrough Dynamic Network Rewiring of DNNs
<p align="center"><img width="30%" src="/images/ASP_DAC_logo.png"></p><br/> 


#### RESNET saved modes:
1. [ResNet18 Channel pruned on CIFAR-10](https://drive.google.com/file/d/1kbyl34OTxt7YBJON6VpdMBNQWejWpGZN/view?usp=sharing)
2. [ResNet18 Irregular pruned on CIFAR-10](https://drive.google.com/file/d/1nmAPjhM0Hlo2I7k6UPeGDta8g2kN6xJI/view?usp=sharing)

#### To test adversarial accuracy of a saved model, please:
1. Copy and bring the model to same location as the *.py files.
2. Open the run_test.py file to change model_type ['resnet18' or 'vgg16'] and provide the dataset='cifar10'.
3. Provide --adv_eval to enable adversarial evaluation.
4. Run python file: 'python run_test.py'
