from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
import os.path as osp
from utils.util import  TwoCropTransform
from torchvision.transforms import InterpolationMode

def moco_cifar_transform(img_size=32):
    transform = transforms.Compose([
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # BYOL ,and reduce strength for CIFAR
                ], p=0.6),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                p = 0.2),
            transforms.RandomResizedCrop(img_size,scale=(0.2, 1.),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    
    return TwoCropTransform(transform)

def moco_cifar_test_transform(img_size=32):
    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    
    return transform
