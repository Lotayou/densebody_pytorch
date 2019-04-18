from data_utils.densebody_dataset import DenseBodyDataset
from data_utils import visualizer as vis
import torch
from models import create_model
from torch.utils.data import DataLoader
import sys 
from sys import platform
import os
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

# default options
def TestOptions(debug=False):
    parser = ArgumentParser()
    
    # dataset options    
    # platform specific options
    windows_root = 'D:/data' 
    linux_root = '/backup1/lingboyang/data'  # change to you dir
    data_root = linux_root if platform == 'linux' else windows_root
    num_threads = 0
    batch_size = 1
    
    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset', type=str, default='human36m',
        choices=['human36m', 'surreal', 'up3d', 'nturgbd'])
    parser.add_argument('--max_dataset_size', type=int, default=-1)
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--name', type=str, default='densebody_resnet_h36m')
    parser.add_argument('--uv_map', type=str, default='radvani', choices=['radvani', 'vbml_close', 'vbml_spaced', 'smpl_fbx'])
    parser.add_argument('--num_threads', default=num_threads, type=int, help='# sthreads for loading data')
    
    # model options
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vggnet', 'mobilenet'])
    parser.add_argument('--netD', type=str, default='convres', choices=['convres', 'conv-up'])
    parser.add_argument('--nz', type=int, default=256, help='latent dims')
    parser.add_argument('--ndown', type=int, default=6, help='downsample times')
    parser.add_argument('--nchannels', type=int, default=64, help='conv channels')
    parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
    parser.add_argument('--nl', type=str, default='relu', choices=['relu', 'lrelu', 'elu'])
    parser.add_argument('--init_type', type=str, default='xavier', choices=['xavier', 'normal', 'kaiming', 'orthogonal'])
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

    # testing options
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--phase', type=str, default='test', choices=['test', 'in_the_wild'])
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--load_epoch', type=int, default=0)
    
    opt = parser.parse_args()
    
    opt.uv_prefix = opt.uv_map + '_template'
    opt.project_root = os.path.dirname(os.path.realpath(__file__))
    
    return opt

if __name__ == '__main__':
    # Change this to your gpu id.
    # The program is fixed to run on a single GPU
    if platform == 'linux':
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    opt = TestOptions(debug=False)
    dataset = DenseBodyDataset(data_root=opt.data_root, dataset_name=opt.dataset, 
            uv_map=opt.uv_map, max_size=opt.max_dataset_size, phase=opt.phase)
    batchs_per_epoch = len(dataset) // opt.batch_size # drop last batch
    print('#testing images = %d' % len(dataset))

    model = create_model(opt)
    model.setup(opt)
    visualizer = vis.Visualizer(opt)
    
    loop = tqdm(range(opt.max_dataset_size), ncols=120)
    for i in loop:
        loop.set_description('Testing case %d' % i)
        data = dataset[i:i+1]
        model.set_input(data)
        with torch.no_grad():
            model.fake_UV = model.decoder(model.encoder(model.real_input))
        
        visualizer.save_results(model.get_current_visuals(), opt.load_epoch, i)
