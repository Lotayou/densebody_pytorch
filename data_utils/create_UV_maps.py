from smpl_torch_batch import SMPLModel
import numpy as np
import pickle 
import h5py
import torch
from torch.nn import Module
import os
import shutil
from tqdm import tqdm
from time import time
from sys import platform
from torch.utils.data import Dataset, DataLoader
from cv2 import imread, imwrite
from skimage.draw import circle

from procrustes import map_3d_to_2d
from uv_map_generator import UV_Map_Generator

class Human36MWashedDataset(Dataset):
    def __init__(self, smpl, max_item=312188, root_dir=None, 
            annotation='h36m.pickle', calc_mesh=False):
        super(Human36MWashedDataset, self).__init__()
        if root_dir is None:            
            root_dir = '/backup1/lingboyang/data/human36m_washed' \
                if platform == 'linux' \
                else 'D:/data/human36m_washed'
            
        self.root_dir = root_dir
        self.calc_mesh = calc_mesh
        self.smpl = smpl
        self.dtype = smpl.data_type
        self.device = smpl.device
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
        with open(root_dir + '/' + annotation, 'rb') as f:
            tmp = pickle.load(f)        
            for k in tmp.keys():
                data = tmp[k][:max_item]
                    
                setattr(self, k, data)
                self.itemlist.append(k)
            
        # flip gt2d y coordinates
        # self.gt2d[:,:,1] = 256 - self.gt2d[:,:,1]
        # # flip gt3d y & z coordinates
        # self.gt3d[:,:,1:] *= -1
        self.length = self.pose.shape[0]
        
    def __getitem__(self, index):
        out_dict = {}
        for item in self.itemlist:
            if item == 'imagename':
                out_dict[item] = getattr(self, item)[index]
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
    
def visualize(folder, imagenames, mesh_2d, joints_2d, root=None):
    i = 0
    for name, mesh, joints in zip(imagenames, mesh_2d, joints_2d):
        print(name)
        shutil.copyfile(root + name,
            '/im_gt_{}.png'.format(folder, i)
        )
        im = imread(name)
        shape = im.shape[0:2]
        height = im.shape[0]
        for p2d in mesh:
            im[height - p2d[1], p2d[0]] = [127,127,127]
            
        for j2d in joints:
            rr, cc = circle(height - j2d[1], j2d[0], 2, shape)
            im[rr, cc] = [255, 0, 0]
            
        imwrite('{}/im_mask_{}.png'.format(folder, i), im)
        i += 1

    
def create_UV_maps(UV_label_root=None, uv_prefix = 'radvani_template'):
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
    dataset = Human36MWashedDataset(model, calc_mesh=True)
    
    generator = UV_Map_Generator(
        UV_height=256,
        UV_pickle=uv_prefix+'.pickle'
    )
    
    # create root folder for UV labels
    if UV_label_root is None:
        UV_label_root=dataset.root_dir.replace('_washed', 
                '_UV_map_{}'.format(uv_prefix[:-9])
    
    if not os.path.isdir(UV_label_root):
        os.makedirs(UV_label_root)
        subs = [sub for sub in os.listdir(dataset.root_dir)
            if os.path.isdir(dataset.root_dir + '/' + sub)]
        for sub in subs:
            os.makedirs(UV_label_root + '/' + sub)
    else:
        print('{} folder exists, process terminated...'.format(UV_label_root))
        return
    
    # generate mesh, align with 14 point ground truth
    batch_size = 64
    total_batch_num = dataset.length // batch_size + 1
    _loop = tqdm(range(total_batch_num), ncols=80)
    
    for batch_id in _loop:
        data = dataset[batch_id * batch_size: (batch_id + 1) * batch_size]
        meshes = data['meshes']
        input = data['lsp_joints']
        target_2d = data['gt2d']
        target_3d = data['gt3d']
        imagename = [UV_label_root + str for str in data['imagename']]
        
        transforms = map_3d_to_2d(input, target_2d, target_3d)
        
        # Important: mesh should be centered at the origin!
        deformed_meshes = transforms(meshes)
        mesh_3d = deformed_meshes.detach().cpu().numpy()
        '''
        test_folder = '_test_radvani'
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)
        visualize(test_folder, data['imagename'], mesh_3d[:,:,:2].astype(np.int), 
           target_2d.detach().cpu().numpy().astype(np.int), dataset.root_dir)
        '''
        s=time()
        for name, mesh in zip(imagename, mesh_3d):
            UV_position_map, verts_backup = \
                generator.get_UV_map(mesh)
            imwrite(name, (UV_position_map * 255).astype(np.uint8))
            
            # write colorized coordinates to ply
            '''
            model.write_obj(
                mesh, 
                '{}/real_mesh_{}.obj'.format(test_folder, i)
            )   # weird.
            UV_scatter, _, _ = generator.render_point_cloud(
                verts=mesh
            )
            
            generator.write_ply(
                '{}/colored_mesh_{}.ply'.format(test_folder, i), mesh
            )
            out = np.concatenate(
                (UV_position_map, UV_scatter), axis=1
            )
            
            imsave('{}/UV_position_map_{}.png'.format(test_folder, i), out)
            
            resampled_mesh = generator.resample(UV_position_map)
            
            model.write_obj(
                resampled_mesh, 
                '{}/recon_mesh_{}.obj'.format(test_folder, i)
            )
            '''

if __name__ == '__main__':
    # Please make sure the prefix is the same as in train.py opt.uv_prefix
    # prefix = 'radvani_template'
    prefix = 'vbml_close_template'
    # prefix = 'vbml_spaced_template'
    create_UV_maps(uv_prefix=prefix)
    # create_UV_maps(uv_prefix='radvani_new_template')
    # create_UV_maps(uv_prefix='smpl_fbx_template')
    
