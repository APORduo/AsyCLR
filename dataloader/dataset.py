import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
DEFAULT_ROOT = '/data'

datasets_dict = {
    'cifar100': 'cifar_cil',
    'mini_imagenet': 'mini_cil',
    'cub200': 'cub_cil'
}

datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    #breakpoint()
    #if kwargs.get('root') is None:
    #    kwargs['root'] = os.path.join(DEFAULT_ROOT, name)
    if name in datasets_dict.keys():
        name = datasets_dict[name]
    dataset = datasets[name](**kwargs)
    return dataset

identity = lambda x:x
class JsonDataset:
  def __init__(self, data_file, transform, target_transform=identity):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])

class FeatureDataset(Dataset):
    def __init__(self,vectors,labels) -> None:
        if isinstance(vectors,np.ndarray):
            self.vectors = torch.from_numpy(vectors)
        
            self.labels = torch.from_numpy(labels)
        else:
            self.vectors = vectors
            self.labels = labels
        
    
    def __getitem__(self,i):
        
        return self.vectors[i],self.labels[i]
    
    def __len__(self):
        return(len(self.labels))
        