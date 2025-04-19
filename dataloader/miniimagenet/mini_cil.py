from PIL import Image
from dataloader.dataset import register
from torch.utils.data import Dataset
import os
import numpy as np
from dataloader.autoaugment_mini import AutoAugImageNetPolicy
from torchvision import transforms
CLASSES = [
    'n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081',
    'n01843383', 'n01855672', 'n01910747', 'n01930112', 'n01981276',
    'n02074367', 'n02089867', 'n02091244', 'n02091831', 'n02099601',
    'n02101006', 'n02105505', 'n02108089', 'n02108551', 'n02108915',
    'n02110063', 'n02110341', 'n02111277', 'n02113712', 'n02114548',
    'n02116738', 'n02120079', 'n02129165', 'n02138441', 'n02165456',
    'n02174001', 'n02219486', 'n02443484', 'n02457408', 'n02606052',
    'n02687172', 'n02747177', 'n02795169', 'n02823428', 'n02871525',
    'n02950826', 'n02966193', 'n02971356', 'n02981792', 'n03017168',
    'n03047690', 'n03062245', 'n03075370', 'n03127925', 'n03146219',
    'n03207743', 'n03220513', 'n03272010', 'n03337140', 'n03347037',
    'n03400231', 'n03417042', 'n03476684', 'n03527444', 'n03535780',
    'n03544143', 'n03584254', 'n03676483', 'n03770439', 'n03773504',
    'n03775546', 'n03838899', 'n03854065', 'n03888605', 'n03908618',
    'n03924679', 'n03980874', 'n03998194', 'n04067472', 'n04146614',
    'n04149813', 'n04243546', 'n04251144', 'n04258138', 'n04275548',
    'n04296562', 'n04389033', 'n04418357', 'n04435653', 'n04443257',
    'n04509417', 'n04515003', 'n04522168', 'n04596742', 'n04604644',
    'n04612504', 'n06794110', 'n07584110', 'n07613480', 'n07697537',
    'n07747607', 'n09246464', 'n09256479', 'n13054560', 'n13133613',
]

FSCIL_SAMPLES = {
    'n03544143': ['00000811.jpg', '00000906.jpg', '00000122.jpg', '00000185.jpg', '00001258.jpg'],
    'n03584254': ['00000818.jpg', '00000891.jpg', '00000145.jpg', '00000186.jpg', '00001254.jpg'],
    'n03676483': ['00000772.jpg', '00000864.jpg', '00000108.jpg', '00000164.jpg', '00001236.jpg'],
    'n03770439': ['00000738.jpg', '00000835.jpg', '00000111.jpg', '00000154.jpg', '00001238.jpg'],
    'n03773504': ['00000781.jpg', '00000880.jpg', '00000132.jpg', '00000177.jpg', '00001237.jpg'],

    'n03775546': ['00000808.jpg', '00000891.jpg', '00000123.jpg', '00000179.jpg', '00001241.jpg'],
    'n03838899': ['00000800.jpg', '00000881.jpg', '00000124.jpg', '00000165.jpg', '00001242.jpg'],
    'n03854065': ['00000796.jpg', '00000899.jpg', '00000103.jpg', '00000146.jpg', '00001249.jpg'],
    'n03888605': ['00000783.jpg', '00000875.jpg', '00000124.jpg', '00000184.jpg', '00001245.jpg'],
    'n03908618': ['00000778.jpg', '00000855.jpg', '00000127.jpg', '00000175.jpg', '00001243.jpg'],

    'n03924679': ['00000785.jpg', '00000871.jpg', '00000115.jpg', '00000164.jpg', '00001240.jpg'],
    'n03980874': ['00000794.jpg', '00000911.jpg', '00000124.jpg', '00000184.jpg', '00001229.jpg'],
    'n03998194': ['00000790.jpg', '00000876.jpg', '00000116.jpg', '00000164.jpg', '00001248.jpg'],
    'n04067472': ['00000802.jpg', '00000904.jpg', '00000130.jpg', '00000175.jpg', '00001257.jpg'],
    'n04146614': ['00000809.jpg', '00000921.jpg', '00000132.jpg', '00000204.jpg', '00001243.jpg'],

    'n04149813': ['00000789.jpg', '00000879.jpg', '00000152.jpg', '00000196.jpg', '00001229.jpg'],
    'n04243546': ['00000785.jpg', '00000868.jpg', '00000109.jpg', '00000159.jpg', '00001243.jpg'],
    'n04251144': ['00000797.jpg', '00000891.jpg', '00000123.jpg', '00000170.jpg', '00001244.jpg'],
    'n04258138': ['00000807.jpg', '00000900.jpg', '00000135.jpg', '00000193.jpg', '00001252.jpg'],
    'n04275548': ['00000755.jpg', '00000854.jpg', '00000127.jpg', '00000168.jpg', '00001238.jpg'],

    'n04296562': ['00000772.jpg', '00000862.jpg', '00000119.jpg', '00000158.jpg', '00001223.jpg'],
    'n04389033': ['00000802.jpg', '00000912.jpg', '00000157.jpg', '00000202.jpg', '00001261.jpg'],
    'n04418357': ['00000746.jpg', '00000848.jpg', '00000111.jpg', '00000163.jpg', '00001226.jpg'],
    'n04435653': ['00000828.jpg', '00000932.jpg', '00000139.jpg', '00000203.jpg', '00001245.jpg'],
    'n04443257': ['00000764.jpg', '00000852.jpg', '00000125.jpg', '00000183.jpg', '00001216.jpg'],

    'n04509417': ['00000801.jpg', '00000882.jpg', '00000149.jpg', '00000201.jpg', '00001242.jpg'],
    'n04515003': ['00000791.jpg', '00000893.jpg', '00000112.jpg', '00000161.jpg', '00001259.jpg'],
    'n04522168': ['00000790.jpg', '00000894.jpg', '00000124.jpg', '00000180.jpg', '00001258.jpg'],
    'n04596742': ['00000809.jpg', '00000897.jpg', '00000132.jpg', '00000189.jpg', '00001241.jpg'],
    'n04604644': ['00000828.jpg', '00000904.jpg', '00000124.jpg', '00000175.jpg', '00001256.jpg'],

    'n04612504': ['00000737.jpg', '00000810.jpg', '00000149.jpg', '00000194.jpg', '00001160.jpg'],
    'n06794110': ['00000773.jpg', '00000882.jpg', '00000124.jpg', '00000199.jpg', '00001256.jpg'],
    'n07584110': ['00000764.jpg', '00000855.jpg', '00000133.jpg', '00000180.jpg', '00001154.jpg'],
    'n07613480': ['00000770.jpg', '00000868.jpg', '00000140.jpg', '00000183.jpg', '00001254.jpg'],
    'n07697537': ['00000774.jpg', '00000862.jpg', '00000142.jpg', '00000181.jpg', '00001231.jpg'],

    'n07747607': ['00000787.jpg', '00000894.jpg', '00000140.jpg', '00000190.jpg', '00001253.jpg'],
    'n09246464': ['00000794.jpg', '00000885.jpg', '00000107.jpg', '00000145.jpg', '00001250.jpg'],
    'n09256479': ['00000823.jpg', '00000899.jpg', '00000154.jpg', '00000208.jpg', '00001246.jpg'],
    'n13054560': ['00000771.jpg', '00000856.jpg', '00000102.jpg', '00000159.jpg', '00001230.jpg'],
    'n13133613': ['00000758.jpg', '00000871.jpg', '00000106.jpg', '00000160.jpg', '00001231.jpg']
}

@register('mini_cil')
class MiniCil(Dataset):
    def __init__(self,root='/data',
                 transform = None,
                 train = True,
                 ) -> None:
        super().__init__()
        self.data_prefix = os.path.join(root,'miniimagenet')
        self.IMAGE_PATH = os.path.join(self.data_prefix,'images')
        self.BASE = CLASSES[:60]
        
        self.data = []
        self.targets = []
        
        
        if train:
            csv_path = os.path.join(self.data_prefix,'train.csv')
            lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
            filename_list = [itm.split(',')[0] for itm in lines]
            cls_list = [itm.split(',')[1] for itm in lines]
            for filename, cls in zip(filename_list, cls_list):
                if cls not in self.BASE:
                    continue # filter out base classes
                
                path = os.path.join(self.IMAGE_PATH, filename)
                label = CLASSES.index(cls)
                self.data.append(path)
                self.targets.append(label)
            
            for cls,filename in FSCIL_SAMPLES.items():
                for itm in filename:
                    path = os.path.join(self.IMAGE_PATH,cls+itm)
                    label = CLASSES.index(cls)
                    self.data.append(path)
                    self.targets.append(label)
            
            self.transform = self.train_transform
        else:
            csv_path = os.path.join(self.data_prefix,'test.csv')
            lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
            filename_list = [itm.split(',')[0] for itm in lines]
            cls_list = [itm.split(',')[1] for itm in lines]
            for filename, cls in zip(filename_list, cls_list):
                path = os.path.join(self.IMAGE_PATH, filename)
                label = CLASSES.index(cls)
                self.data.append(path)
                self.targets.append(label)
            
            self.transform = self.test_transform
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    def __getitem__(self, i ):
        
        path, label = self.data[i], self.targets[i]
        img= Image.open(path).convert('RGB')
        img = self.transform(img)
        return img,label
        
        
    
    @property
    def weak_transform(self):
        transform = transforms.Compose([
                transforms.RandomResizedCrop(84),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        return transform   
    @property
    def train_transform(self):
        transform = transforms.Compose([
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(),
                AutoAugImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        return transform  

    @property
    def test_transform(self):
        transform = transforms.Compose([
                transforms.Resize([92,92]),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        return transform  
                
    
    def __len__(self,):
        return len(self.data)
            
    
if __name__ == '__main__':
    #python -m dataloader.miniimagenet.mini_cil
    mini = MiniCil(train=False)
    base_index = [i for i in range(len(mini)) if mini.targets[i] < 60]
    base_data = mini.data[base_index]
    base_target = mini.targets[base_index]
    base_name = [CLASSES[i] for i in range(60)]

    