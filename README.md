
## DNR: A Tunable Robust Pruning FrameworkThrough Dynamic Network Rewiring of DNNs
<p align="center"><img width="30%" src="/images/ASP_DAC_logo.png"></p><br/> 


#### RESNET saved modes:
[ResNet18 on CIFAR-10] (https://drive.google.com/drive/u/2/folders/1q5CKftAQO_zo3PhuZJ07XyjmwFqwqXGX)

#### To test adversarial accuracy of a saved model, please:
1. Copy and bring the model to same location as the *.py files.
2. Open the run_test.py file to change model_type ['resnet18' or 'vgg16'] and provide the dataset='cifar10'.
3. Provide --adv_eval to enable adversarial evaluation.
4. Run python file: 'python run_test.py'
