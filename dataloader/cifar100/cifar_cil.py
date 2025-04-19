from typing import Callable, Optional
from torchvision.datasets.cifar import CIFAR100
from dataloader.dataset import register
from typing import Any, Callable, Optional, Tuple
from dataloader.autoaugment_cifar import CIFAR10Policy
from torchvision import transforms
from PIL import Image
import numpy as np
CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',

    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
FSCIL_SAMPLES = {
    'plain': [29774, 33344, 4815, 6772, 48317],
    'plate': [29918, 33262, 5138, 7342, 47874],
    'poppy': [28864, 32471, 4316, 6436, 47498],
    'porcupine': [29802, 33159, 3730, 5093, 47740],
    'possum': [30548, 34549, 2845, 4996, 47866],

    'rabbit': [28855, 32834, 4603, 6914, 48126],
    'raccoon': [29932, 33300, 3860, 5424, 47055],
    'ray': [29434, 32604, 4609, 6380, 47844],
    'road': [30456, 34217, 4361, 6550, 46896],
    'rocket': [29664, 32857, 4923, 7502, 47270],

    'rose': [31267, 34427, 4799, 6611, 47404],
    'sea': [28509, 31687, 3477, 5563, 48003],
    'seal': [29545, 33412, 5114, 6808, 47692],
    'shark': [29209, 33265, 4131, 6401, 48102],
    'shrew': [31290, 34432, 6060, 8451, 48279],

    'skunk': [32337, 35646, 6022, 9048, 48584],
    'skyscraper': [30768, 34394, 5091, 6510, 48023],
    'snail': [30310, 33230, 5098, 6671, 48349],
    'snake': [29690, 33490, 4260, 5916, 47371],
    'spider': [31173, 34943, 4517, 6494, 47689],

    'squirrel': [30281, 33894, 3768, 6113, 48095],
    'streetcar': [28913, 32821, 6172, 8276, 48004],
    'sunflower': [31249, 34088, 5257, 6961, 47534],
    'sweet_pepper': [30404, 34101, 4985, 6899, 48115],
    'table': [31823, 35148, 3922, 6548, 48127],

    'tank': [30815, 34450, 3481, 5089, 47913],
    'telephone': [31683, 34591, 5251, 7608, 47984],
    'television': [29837, 33823, 4615, 6448, 47752],
    'tiger': [31222, 34079, 5686, 7919, 48675],
    'tractor': [28567, 32964, 5009, 6201, 47039],

    'train': [29355, 33909, 3982, 5389, 47166],
    'trout': [31058, 35180, 5177, 6890, 48032],
    'tulip': [31176, 35098, 5235, 7861, 47830],
    'turtle': [30874, 34639, 5266, 7489, 47323],
    'wardrobe': [29960, 34050, 4988, 7434, 48208],

    'whale': [30463, 34580, 5230, 6813, 48605],
    'willow_tree': [31702, 35249, 5854, 7765, 48444],
    'wolf': [30380, 34028, 5211, 7433, 47988],
    'woman': [31348, 34021, 4929, 7033, 47904],
    'worm': [30627, 33728, 4895, 6299, 47507],
}

@register('cifar_cil')
class CifarCil(CIFAR100):
    def __init__(self, root: str='/data', train: bool = True, 
                 transform: Callable=None,
                 target_transform: Callable[..., Any]=None , 
                 download: bool = False) :
        super().__init__(root, train, transform, target_transform, download)
        
        base_index = [i for i in range(len(self.targets)) if self.targets[i] in range(0, 60)]
        train_index = base_index.copy()
        for k ,v in FSCIL_SAMPLES.items():
            for i in v:
                train_index.append(i)
        
        self.train_transform = transforms.Compose([
              
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        if train:
            self.data = self.data[train_index]
            self.targets = np.array(self.targets)
            self.targets = self.targets[train_index]
           
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform
            
        
        
        def __getitem__(self, index):
            img_o, target = self.data[index], self.targets[index]
            img_o = Image.fromarray(img_o)

            if self.transform is not None:
                img_o = self.transform(img_o)

            if self.target_transform is not None:
                target = self.target_transform(target)

    
            return img_o, target

if __name__ == '__main__':
    cifarcil = CifarCil()
    cifarcil_test = CifarCil(train=False)
    #breakpoint()