import numpy as np
import torch
from cv2 import imread
from torch.utils.data import Dataset
from sys import platform
import pickle
import os
from torchvision import transforms
from PIL import Image

# TODO: Change the global directory to where you normally hold the datasets.
# I use both Windows PC and Linux Server for this project so I have two dirs.

windows_root = 'D:/data'
linux_root = '/backup1/lingboyang/data'
data_root = linux_root if platform == 'linux' else windows_root

im_trans = transforms.Compose([
    Image.fromarray,
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])

'''
    DenseBodyDataset: Return paired human image and UV position map, 
    along with ground truth 3D and 2D skeleton annotations.
    
    All human pics are stored in folder '{path_to_your_dataset}/{dataset_name}_washed'
        along with a processed h5 file containing necessary annotations.
    All UV maps are stored in folder '{path_to_your_dataset}/{dataset_name}_UV_map'
    
    Note: In the annotation pickle, all image paths are relative
    
'''
class DenseBodyDataset(Dataset):
    def __init__(self, data_root=data_root, uv_map='radvani', dataset_name='human36m', 
        annotation = 'h36m.pickle', phase='train', train_test_split=0.8, max_size=-1, device=None, transform=im_trans):
        
        super(DenseBodyDataset, self).__init__()
        self.im_root = '{}/{}_washed'.format(data_root, dataset_name)
        self.uv_root = '{}/{}_UV_map_{}'.format(data_root, dataset_name, uv_map)
        
        if not os.path.isdir(self.im_root):
            raise(FileNotFoundError('{} dataset not found, '.format(dataset_name) + 
                'please run "data_washing.py" first'))
        if not os.path.isdir(self.uv_root):
            raise(FileNotFoundError('{} uv map not found, '.format(uv_map) + 
                'please run "create_dataset.py" first'))
        
        # parse annotation
        self.itemlist = []
        with open(self.im_root + '/' + annotation, 'rb') as f:
            tmp = pickle.load(f)
            
        # Prepare train/test split
        total_length = tmp['pose'].shape[0]
        split_point = int(train_test_split * total_length)
        for k in tmp.keys():
            data = tmp[k]
            if phase == 'train':
                data = data[0:split_point]
                cur_length = split_point
            elif phase == 'test':
                data = data[split_point:total_length]
                cur_length = total_length - split_point
            
            if 0 < max_size < cur_length:
                data = data[:max_size]
            
            if k == 'imagename':
                self.im_names = [self.im_root + s for s in data]
                self.uv_names = [self.uv_root + s for s in data]
                self.itemlist.append('im_names')
                self.itemlist.append('uv_names')
            else:
                setattr(self, k, data)
                self.itemlist.append(k)
        
        self.length = self.pose.shape[0]
        # Set image transforms and device
        self.device = torch.device('cuda') if device is None else device
        self.transform = transform
    
    def __getitem__(self, id):
        out_dict = {}
        for k in self.itemlist:
            items = getattr(self, k)[id]
            if k.endswith('names'):
                ims = [self.transform(imread(item)) for item in items]
                out_dict[k.replace('names','data')] = torch.stack(ims).to(self.device)
            else:
                out_dict[k] = torch.from_numpy(items)
        return out_dict    
        
    def __len__(self):
        return self.length

if __name__ == '__main__':
    '''
    Return sample:
        gt2d cpu torch.int64 torch.Size([10, 14, 2]) tensor(225)
        gt3d cpu torch.float64 torch.Size([10, 14, 3]) tensor(0.7802, dtype=torch.float64)
        im_ cuda:0 torch.float32 torch.Size([10, 3, 256, 256]) tensor(1., device='cuda:0')
        uv_ cuda:0 torch.float32 torch.Size([10, 3, 256, 256]) tensor(1., device='cuda:0')
        pose cpu torch.float64 torch.Size([10, 72]) tensor(2.9966, dtype=torch.float64)
        shape cpu torch.float64 torch.Size([10, 10]) tensor(2.1988, dtype=torch.float64)
    '''
    dataset = DenseBodyDataset()
    print(len(dataset), dataset.itemlist)
    data = dataset[0:10]
    for k, v in data.items():
        print(k, v.device, v.dtype, v.shape, v.max())
    