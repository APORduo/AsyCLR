import numpy as np
import torch
import random
from math import sqrt

from torchvision import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader,Dataset

from dataloader.miniimagenet.miniimagenet import MiniImageNet
import pdb

#__all__= ['get_dataloader','set_up_datasets','get_session_test_loader','get_session_classes']
def set_up_datasets(args):

    if 'mini' in args.dataset.lower():
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.dataset = 'mini_imagenet'
        args.base_class = 60
        args.num_classes = 100
        args.n_way = 5
        args.n_shot = 5
        args.n_sessions = 9
    else:
        Exception('Undefined dataset name!')
    return args


def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args,session)
    return trainset, trainloader, testloader


def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    trainset = None

    if args.dataset == 'mini_imagenet'  :
        trainset = MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = MiniImageNet(root=args.dataroot, train=False, index=class_index)

    
    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def get_class_dataloader(args,class_index):
    if isinstance(class_index, int):
        class_index = [class_index]
    

    if args.dataset == 'mini_imagenet':
        trainset = MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = MiniImageNet(root=args.dataroot, train=False, index=class_index)

 
    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

        
    

def get_new_dataloader(args,session):
    txt_path = "./index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'

    if args.dataset == 'mini_imagenet':
        trainset = MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
        
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True, # shuffle=False
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)
    testset = MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_test_loader(args,session):
    class_new = get_session_classes(args, session)
    if args.dataset == 'mini_imagenet':
        testset = MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = DataLoader(dataset=testset, batch_size=args.batch_size_test, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return testloader
    

def get_session_classes(args,session):
    class_list = np.arange(args.base_class + session * args.n_way)
    return class_list


class FeatureDataset(Dataset):
    def __init__(self,vectors,labels) -> None:
        self.vectors = torch.from_numpy(vectors) 
        self.labels = torch.from_numpy(labels)
        
    
    def __getitem__(self,i):
        
        return self.vectors[i],self.labels[i]
    
    def __len__(self):
        return(len(self.labels))
        

class InfiniteDataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

    def __iter__(self):
        return self