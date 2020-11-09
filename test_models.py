from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import copy

from models import VGG
from ResNet import ResNet18
from attack_model import Attack
from normalization_layer import Normalize_layer

from torchvision import transforms
import torchvision.datasets as dset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate(args, model, device, criterion, test_loader, is_test_set=False, attacker=None, adv_eval=False):
    model.eval()
    test_loss = 0
    correct = 0
    correct_pgd = 0
    correct_fgsm = 0
    total = 0
    
    for i, (inputs, target) in enumerate(test_loader):
        inputs, target = inputs.to(device), target.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if adv_eval and (attacker is not None):
            inputs.requires_grad = False
            attacker.update_params(attack_method='fgsm')
            perturbed_data = attacker.attack_method(model, inputs, target)
            output_fgsm = model(perturbed_data)
            loss_fgsm = criterion(output_fgsm, target)
            _, predicted_fgsm = output_fgsm.max(1)
            correct_fgsm += predicted_fgsm.eq(target).sum().item()
            
            inputs.requires_grad = False
            attacker.update_params(attack_method='pgd')
            perturbed_data = attacker.attack_method(model, inputs, target)
            output_pgd = model(perturbed_data)
            loss_pgd = criterion(output_pgd, target)
            _, predicted_pgd = output_pgd.max(1)
            correct_pgd += predicted_pgd.eq(target).sum().item()
            
    acc_cleanTest = 100.*correct/total
    acc_fgsmTest = 100.*correct_fgsm/total
    acc_pgdTest = 100.*correct_pgd/total
            
    print('Clean acc:{}, FGSM acc:{}, PGD acc:{}'.format(acc_cleanTest, acc_fgsmTest, acc_pgdTest))
            
    
def main():
    parser = argparse.ArgumentParser(description='Pytorch MNIST example')
    
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--adv_eval', dest='adv_eval', action='store_true',
                    help='enable the adversarial evaluation')
    parser.add_argument('--model_type', type=str, default='vgg16',
                    help = 'the models to be used for training [vgg16, resnet18,\
                    resnet50]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='the dataset to be used for training [cifar10, cifar100]')
    parser.add_argument('--comp_admm', dest='comp_admm', action='store_true',
                    help='enable the comparison model saved from admm_based_robust pruning')
    parser.add_argument('--comp_l1Lasso', dest='comp_l1Lasso', action='store_true',
                    help='enable the comparison model saved from l1Lasso_based_robust pruning')
    
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        # Current data-preprocessing does not include the normalization    
        normal_train_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()]
        
        normal_test_transform = [
        transforms.ToTensor()]
        
        if not (args.adv_eval):
            normal_train_transform.append(transforms.Normalize(mean, std))
            normal_test_transform.append(transforms.Normalize(mean, std))
        
        train_transform = transforms.Compose(normal_train_transform)
        test_transform = transforms.Compose(normal_test_transform)
                
        train_data = dset.CIFAR10(
            './data', train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            './data', train=False, transform=test_transform, download=True)
        num_classes = 10
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    ##########################################################
    ## Assign the model for testing the accuracies
    ##########################################################
    if args.model_type == 'resnet18' and (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        net_c = ResNet18(num_classes).to(device)
        
    if args.model_type == 'vgg16' and (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        net_c = VGG('VGG16', init_weights = True).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    net = torch.nn.Sequential(
                    Normalize_layer(mean, std),
                    net_c
                    )
    ##########################################################
    ## Load the pre-trained models for testing the accuracies
    ##########################################################
    if args.model_type == 'vgg16' and args.dataset == 'cifar10':
        net.load_state_dict(torch.load\
    	("vgg16_cifar10_dens0.05_magnitude_epoch200_testAcc_86.73999786376953.pt"))
        net.to(device)
    elif args.model_type == 'resnet18' and args.dataset == 'cifar10':
        net.load_state_dict(torch.load\
         ("resnet18_cifar10_dens0.05_magnitude_epoch200_testAcc_87.31999969482422.pt"))
        net.to(device)

    model_attack = Attack(dataloader=train_loader,
                          attack_method='pgd', epsilon=0.031) 	
    
    print('------------------------------------------------\n')
    print('Reporting results on model: {} for dataset:{}'.format(args.model_type, args.dataset))
    print('------------------------------------------------\n')
    validate(args, net, device, criterion, test_loader,\
    	is_test_set=True, attacker=model_attack, adv_eval=args.adv_eval)
    print('------------------------------------------------\n')
if __name__ == '__main__':
    main()


