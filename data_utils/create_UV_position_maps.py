from smpl_torch_batch import SMPLModel
import numpy as np
import pickle 
import h5py
import torch
from torch.nn import Module
import os
import shutil
from sys import platform
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
from skimage.draw import circle

from procrustes import map_3d_to_2d
from save_texture_map import get_UV_position_map as get_UV


class Human36MDataset(Dataset):
    def __init__(self, smpl, max_item=312188, root_dir=None, 
            annotation='annotation_large.h5', calc_mesh=False):
        super(Human36MDataset, self).__init__()
        if root_dir is None:            
            root_dir = '/backup1/lingboyang/data/human36m' \
                if platform == 'linux' \
                else 'D:/data/human36m'
            
        self.root_dir = root_dir
        self.calc_mesh = calc_mesh
        self.smpl = smpl
        self.dtype = smpl.data_type
        self.device = smpl.device
        fin = h5py.File(os.path.join(root_dir, annotation), 'r')
        self.itemlist = []
        '''
        center
        gt2d
        gt3d
        height
        imagename
        pose
        shape
        smpl_joint
        width
        '''
        for k in fin.keys():
            data = fin[k][:max_item]
            if k.startswith('gt'):  # reshape coords
                data = data.reshape(-1, 14, 3)
                if k == 'gt2d':  # remove confidence score
                    data = data[:,:,:2]
                
            setattr(self, k, data)
            self.itemlist.append(k)
        
        # flip gt2d y coordinates
        self.gt2d[:,:,1] = self.height[:, np.newaxis] - self.gt2d[:,:,1]
        # flip gt3d y & z coordinates
        self.gt3d[:,:,1:] *= -1
        
        
        fin.close()
        self.length = self.pose.shape[0]
        
    def __getitem__(self, index):
        out_dict = {}
        for item in self.itemlist:
            if item == 'imagename':
                out_dict[item] = [self.root_dir + '/' + b.decode()
                    for b in getattr(self, item)[index]]
            else:
                data_npy = getattr(self, item)[index]
                out_dict[item] = torch.from_numpy(data_npy)\
                    .type(self.dtype).to(self.device)
            
        if self.calc_mesh:
            _trans = torch.zeros((out_dict['pose'].shape[0], 3), 
                dtype=self.dtype, device=self.device)
            meshes, lsp_joints = self.smpl(out_dict['shape'], out_dict['pose'], _trans)
            out_dict['meshes'] = meshes
            out_dict['lsp_joints'] = lsp_joints
        
        return out_dict
        
    def __len__(self):
        return self.length
    

def visualize(imagenames, mesh_2d, joints_2d):
    i = 0
    for name, mesh, joints in zip(imagenames, mesh_2d, joints_2d):
        shutil.copyfile(name,
            '_test_cache/im_gt_{}.png'.format(i)
        )
        im = imread(name)
        shape = im.shape[0:2]
        height = im.shape[0]
        for p2d in mesh:
            im[height - p2d[1], p2d[0]] = [127,127,127]
            
        for j2d in joints:
            rr, cc = circle(height - j2d[1], j2d[0], 2, shape)
            im[rr, cc] = [255, 0, 0]
            
        imsave('_test_cache/im_mask_{}.png'.format(i), im)
        i += 1

    
def run_test():
    if platform == 'linux':
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    data_type = torch.float32
    device=torch.device('cuda')
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    model = SMPLModel(
            device=device,
            model_path = './model_lsp.pkl',
            data_type=data_type,
        )
    dataset = Human36MDataset(model, max_item=100, calc_mesh=True)
    
    # generate mesh, align with 14 point ground truth
    case_num = 10
    data = dataset[:case_num]
    meshes = data['meshes']
    input = data['lsp_joints']
    target_2d = data['gt2d']
    target_3d = data['gt3d']
    
    transforms = map_3d_to_2d(input, target_2d, target_3d)
    
    # Important: mesh should be centered at the origin!
    deformed_meshes = transforms(meshes)
    mesh_3d = deformed_meshes.detach().cpu().numpy().astype(np.int)
    #visualize(data['imagename'], mesh_3d[:,:,:2], target_2d.detach().cpu().numpy().astype(np.int))
    
    for i, mesh in enumerate(mesh_3d):
        img = get_UV(mesh, 300)
        imsave('_test_cache/UV_map_{}.png'.format(i), img)
    
if __name__ == '__main__':
    run_test()
