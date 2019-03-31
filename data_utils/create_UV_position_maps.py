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

if platform == 'linux':
    from batch_svd import batch_svd

class Human36MDataset(Dataset):
    def __init__(self, smpl, max_item=312188, root_dir=None, 
            annotation='annotation_large.h5', calc_mesh=False):
        super(Human36MDataset, self).__init__()
        if root_dir is None:
            root_dir = 'D:/data/human36m'
            
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
    
'''
    procrustes_3d_to_2d: 
        Align 3D input keypoints to 2D/3D ground truth via Procrustes analysis
        
    Parameters:
    ------------------------------------------------------------
    J3d: [N * J * 3]: batch 3D input 
    gt3d: groung truth 3D input
    gt2d: groung truth 2D input
    
    Output:
    ------------------------------------------------------------
    A transformation function that directly takes input N * None * 3 points
    and align it with 2D annotations.
    
    Algorithm:
    ------------------------------------------------------------
    @ Using ground truth 3D to determine rotation
    @ Using ground truth 2D to determine translation and scaling
'''    
def procrustes_3d_to_2d(J3d, gt2d, gt3d):
    batch_size = J3d.shape[0]
    
    # Part 1: Align J3d with gt3d
    
    ### i. centering: must
    G3d = gt3d.clone()
    cent_J = torch.mean(J3d, dim=1, keepdim=True)
    J3d -= cent_J
    cent_G = torch.mean(G3d, dim=1, keepdim=True)
    G3d -= cent_G
    
    ### ii. scaling: not necessary here.
    
    ### iii. rotation
    M = torch.bmm(G3d.transpose(1,2), J3d) # [N, 3, 3]
    if platform == 'linux':
        U, D, V = batch_svd(M)
        R = torch.bmm(V, U.transpose(1,2))
    else:
        R = [None] * batch_size
        for i in range(batch_size):
            U, D, V = torch.svd(M[i])
            R[i] = torch.mm(V, U.transpose(0,1)) # transpose
        R = torch.stack(R, dim=0)
        
    reg3d = torch.bmm(J3d, R)
    ### eval stage I: rel error < 5%
    '''
    print(reg3d - G3d)
    test_case = range(10)
    for i in test_case:
        #np.savetxt('_test_cache/2d_gt_{}.xyz'.format(i), gt2d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/3d_in_{}.xyz'.format(i), J3d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/3d_reg_{}.xyz'.format(i), reg3d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/3d_gt_{}.xyz'.format(i), G3d[i].detach().cpu().numpy(), delimiter=' ')
    '''
    
    # Part II: Align reg3d with gt2d
    G2d = gt2d.clone()
    reg2d = reg3d[:, :, :2]
    ### i. translation
    cent2d = torch.mean(G2d, dim=1, keepdim=True)
    G2d -= cent2d
    cent3d = torch.cat((cent2d,
        torch.zeros((batch_size, 1, 1), dtype=cent2d.dtype, device=cent2d.device)
    ), dim=2)
    
    ### ii. scaling
    G2d = G2d.view(batch_size, -1)
    r2d = reg2d.reshape(batch_size, -1)
    s = torch.sum(r2d * G2d, dim=1) / torch.sum(r2d * r2d, dim=1)
    s = s.unsqueeze(dim=1).unsqueeze(dim=2)
    
    ### eval: max 5 pixels error...
    '''
    reg2d = reg2d * s + cent2d
    print(reg2d - gt2d)
    # eval: make sure joint locations matching
    # 2D not very accurate, consider 3D?
    # gt3d and gt2d spatially aligned, roughly s * gt3d[:,:,:2] + t = gt2d
    
    gt2d = torch.cat((gt2d,
        torch.zeros((batch_size, gt2d.shape[1], 1), dtype=gt2d.dtype, device=gt2d.device)
    ), dim=2)
    reg2d = torch.cat((reg2d,
        torch.zeros((batch_size, reg2d.shape[1], 1), dtype=reg2d.dtype, device=reg2d.device)
    ), dim=2)
    
    test_case = range(10)
    for i in test_case:
        np.savetxt('_test_cache/2d_gt_{}.xyz'.format(i), gt2d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/2d_reg_{}.xyz'.format(i), reg2d[i].detach().cpu().numpy(), delimiter=' ')
    '''
    
    # Wrap-up the result into a function, keep z component as a depth inidcator
    def transform(x):
        return torch.bmm(x - cent_J, R) * s + cent3d
        
    return transform
        
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
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
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
    
    transforms = procrustes_3d_to_2d(input, target_2d, target_3d)
    
    # Important: mesh should be centered at the origin!
    deformed_meshes = transforms(meshes)
    mesh_2d = deformed_meshes.detach().cpu().numpy().astype(np.int)[:,:,:2]
    visualize(data['imagename'], mesh_2d, target_2d.detach().cpu().numpy().astype(np.int))
    
    
if __name__ == '__main__':
    run_test()
