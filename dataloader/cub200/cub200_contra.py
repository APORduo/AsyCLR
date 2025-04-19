from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
import os.path as osp
from utils.util import  TwoCropTransform

def moco_cub_transform():
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                p = 0.2),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                np.array([0.2726, 0.2634, 0.2794]))
            ])
    return TwoCropTransform(transform)
