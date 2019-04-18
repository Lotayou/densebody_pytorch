from .smpl_torch_batch import SMPLModel
from .uv_map_generator import UV_Map_Generator
import os
from cv2 import imwrite
import torch
import numpy as np

class Visualizer():
    def __init__(self, opt):
        os.chdir(opt.project_root + '/data_utils')
        self.UV_sampler = UV_Map_Generator(
            UV_height=opt.im_size,
            UV_pickle=opt.uv_prefix+'.pickle'
        )
        # Only use save obj 
        self.model = SMPLModel(
            device=None,
            model_path = './model_lsp.pkl',
        )
        os.chdir(opt.project_root)
        if opt.phase == 'train':
            self.save_root = '{}/{}/visuals/'.format(opt.checkpoints_dir, opt.name)
        else:
            self.save_root = '{}/{}/visuals/'.format(opt.results_dir, opt.name)
        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)
    
    @staticmethod
    def tensor2im(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy uint8 [0,255] (HWC)
        return ((tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) * 127.5).astype(np.uint8)
    
    @staticmethod    
    def tensor2numpy(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy float [0,1] (HWC)
        return (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) / 2.
    
    def save_results(self, visual_dict, epoch, batch):
        img_name = self.save_root + '{:03d}_{:05d}.png'.format(epoch, batch)
        obj_name = self.save_root + '{:03d}_{:05d}.obj'.format(epoch, batch)
        ply_name = self.save_root + '{:03d}_{:05d}.ply'.format(epoch, batch)
        imwrite(img_name, 
            self.tensor2im(torch.cat([im for im in visual_dict.values()], dim=2))
        )
        fake_UV = visual_dict['fake_UV']
        resampled_verts = self.UV_sampler.resample(self.tensor2numpy(fake_UV))
        self.UV_sampler.write_ply(ply_name, resampled_verts)
        self.model.write_obj(resampled_verts, obj_name)
            
