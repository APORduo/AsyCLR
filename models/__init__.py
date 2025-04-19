from .resnet20_cifar import resnet20
from .resnet12 import ResNet12
from .resnet18_encoder import resnet18
from functools import partial
model_dict = {
    "resnet18":resnet18,
    "resnet18_cub":  partial(resnet18,pretrained = True),
    "resnet20":resnet20,
    "resnet12":ResNet12, 
}