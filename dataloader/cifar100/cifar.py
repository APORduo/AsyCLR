import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import random
import sys

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from dataloader.autoaugment_cifar import CIFAR10Policy, Cutout
from dataloader.trans import get_pre_transform,colorful_spectrum_mix

#from dataloader import register
class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root='/home/dl/llzhao/CEC-CVPR2021/data', 
                 train=True, transform=None, target_transform=None,
                 download=False, index=None, base_sess=None):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.return_idx = False

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
   

        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list
        self.train_transform = transforms.Compose([
              
                transforms.RandomCrop(32, padding=4),
                #transforms.RandomResizedCrop(32, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomApply([CIFAR10Policy()]),    # add AutoAug
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=8),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
         
        
        self.test_transform = transforms.Compose([
              # # transforms.Resize((36,36), interpolation=InterpolationMode.BICUBIC),
                #transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        if self.train:
            downloaded_list = self.train_list
            self.transform = self.train_transform
            
        else:
            downloaded_list = self.test_list
            self.transform = self.test_transform
            # self.transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            # ])

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = np.asarray(self.targets)

        if base_sess:
            self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
        else:  # new Class session
            if train:
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

        self._load_meta()

    def SelectfromDefault(self, data, targets, index):
        data_o = []
        targets_o = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if len(data_o) == 0:
                data_o = data[ind_cl]
                targets_o = targets[ind_cl]
            else:
                data_o = np.vstack((data_o, data[ind_cl]))
                targets_o = np.hstack((targets_o, targets[ind_cl]))

        return data_o, targets_o

    def NewClassSelector(self, data, targets, index):
        data_o = []
        targets_o = []
        ind_list = [int(i) for i in index]
        data_o = data[ind_list]
        targets_o = targets[ind_list]

        return data_o, targets_o
        ind_np = np.array(ind_list)
        index = ind_np.reshape((5,5))
        for i in index:
            ind_cl = i
            if len(data_o) ==0:
                data_o = data[ind_cl]
                targets_o = targets[ind_cl]
            else:
                data_o = np.vstack((data_o, data[ind_cl]))
                targets_o = np.hstack((targets_o, targets[ind_cl]))

        return data_o, targets_o

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
   
        img_o, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_o = Image.fromarray(img_o)

        if self.transform is not None:
            img_o = self.transform(img_o)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img_o, target, index
        else: 
            return img_o, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    
    @property
    def fine_transform(self):
        transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
        return transform

#@register('cifar-cil')
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class FourierCifar(CIFAR100):
    def __init__(self, root='/home/dl/llzhao/CEC-CVPR2021/data', train=True, transform=None, target_transform=None, download=False, index=None, base_sess=False):
        super().__init__(root, train, transform, target_transform, download, index, base_sess)
        self.transform  = get_pre_transform(image_size=84,jitter=0.4)
        self.post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        
    def __getitem__(self, index):
        data_o =  super().__getitem__(index)    
        img_o= data_o[0]
        tag_o = data_o[1]
        assert isinstance(img_o,np.ndarray)
        idx_s = self.sample_image(tag_o,index)[0]
        data_s =  super().__getitem__(idx_s)    
        img_s = data_s[0]
        img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=1)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
        img = [img_o, img_s, img_s2o, img_o2s]
        label = [tag_o, tag_o, tag_o, tag_o]
        
        
        return img,label
        
    
    def sample_image(self,tag,idx):
        
        target_index = (self.targets == tag).nonzero()[0].tolist()
        target_index.remove(idx)
        idx2 = random.sample(target_index,1)
        
        return idx2        
        
        
if __name__ == "__main__":
    dataroot = '/data/'
    batch_size_base = 128
    txt_path = "/home/lduo/code/fewshot/CEC-CVPR2021/data/index_list/cifar100/session_2.txt"
    class_index = np.arange(60)
    trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
                        base_sess=True)
    testset = CIFAR100(root=dataroot, train=False, download=False,index=class_index, base_sess=True)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
    print(testloader.dataset.data.shape)
    
    class_index = open(txt_path).read().splitlines()
    fcifar = FourierCifar(root="/data",train=True,index=class_index,base_sess=False)
    print(len(fcifar[0]))

