import numpy as np
import pickle 
import h5py
import torch
from torch.nn import Module
import os
import shutil
from sys import platform
#from skimage.io import imread, imsave
from cv2 import imread, imwrite
from skimage.transform import resize
from skimage.draw import circle
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class DataWasher():
    def __init__(self, data_type=torch.float32, max_item=312188, root_dir=None, 
            annotation='annotation_large.h5', out_im_size=256):
        super(DataWasher, self).__init__()
        if root_dir is None:            
            root_dir = '/backup1/lingboyang/data/human36m' \
                if platform == 'linux' \
                else 'D:/data/human36m'
            
        self.root_dir = root_dir
        self.dtype = data_type
        self.out_size = out_im_size
        self.itemlist = []
        fin = h5py.File(root_dir + '/' + annotation, 'r')
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
            elif k == 'imagename':
                data = [b.decode() for b in data]
                
            setattr(self, k, data)
            self.itemlist.append(k)
        
        # flip gt2d y coordinates
        self.gt2d[:,:,1] = self.height[:, np.newaxis] - self.gt2d[:,:,1]
        # flip gt3d y & z coordinates
        self.gt3d[:,:,1:] *= -1
        
        fin.close()
        self._strip_non_square_images()
        
    def _strip_non_square_images(self):
        length = self.pose.shape[0]
        is_valid = np.zeros(length)
        for i in range(length):
            if self.height[i] > self.out_size \
            and self.width[i] > self.out_size:
                is_valid[i] = 1
            #else:
            #    print('Image {} invalid: h={}, w={}'\
            #        .format(i,self.height[i], self.width[i]))
        
        valid_indices = np.nonzero(is_valid)[0]
        for k in self.itemlist:
            if k == 'imagename':
                self.imagename = [
                    self.imagename[i]
                    for i in valid_indices
                ]
            else:
                data = getattr(self, k)
                setattr(self, k, data[valid_indices])
                
        self.length = self.pose.shape[0]
    
    def _rand_crop_and_resize(self, img, i, crop_margin=40):
        h = img.shape[0]
        w = img.shape[1]
        # random_crop and resize
        x2d = self.gt2d[i, :, 0]
        y2d = self.gt2d[i, :, 1]
        w_span = x2d.max() - x2d.min() + crop_margin
        h_span = y2d.max() - y2d.min() + crop_margin
        
        crop_size = np.random.randint(
            low=max(h_span, w_span), 
            high=min(h, w)-6
        )
        top = (max(0, y2d.max() - crop_size + 20) + 
            min(y2d.min() - 20, h - crop_size)) // 2
        
        # top = np.random.randint(
            # low=max(0, y2d.max() - crop_size + 20),
            # high=min(y2d.min() - 20, h - crop_size)
        # )
        
        left = (max(0, x2d.max() - crop_size + 20) + 
            min(x2d.min() - 20, w - crop_size) ) // 2
        
        # left = np.random.randint(
            # low=max(0, x2d.max() - crop_size + 20),
            # high=min(x2d.min() - 20, w - crop_size)
        # )
        
        img = img[top:top+crop_size, left:left+crop_size, :]
        img = resize(img, (self.out_size, self.out_size))
        factor = self.out_size / crop_size

        self.gt2d[i, :, 0] = (self.gt2d[i, :, 0] - left) * factor
        self.gt2d[i, :, 1] = (self.gt2d[i, :, 1] + top - h + crop_size) * factor

        return img
    
    def _lrflip(self, img, i):
        img = img[:, ::-1, :]
        self.gt2d[i, :, 0] = self.out_size - self.gt2d[i, :, 0]
        self.gt3d[i, :, 0] *= -1.
        return img
    
    def _visualize(self, img_name, im, joints):
        shape = im.shape[0:2]
        height = im.shape[0]
        # for p2d in mesh:
        #    im[height - p2d[1], p2d[0]] = [127,127,127]
            
        for j2d in joints:
            rr, cc = circle(height - j2d[1], j2d[0], 2, shape)
            im[rr, cc] = [1., 0., 0.]
            
        imsave(img_name, im)
        
    # add Gaussian noise
    def _add_noise(self, img, sigma):
        img += np.random.standard_normal(img.shape) * sigma
        img = np.maximum(0., np.minimum(1., img))
        return img
        
    def data_augmentation(self, target_folder=None, 
        annotation_pickle='h36m.pickle', sigma_noise=0.02):
        # for each item, open and perform random augmentation
        # then save the corrected image in new place.
        if target_folder is None or target_folder == self.root_dir:
            print('Warning: target_folder is invalid, using default name')
            target_folder = self.root_dir + '_washed'
        
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
            subs = [sub for sub in os.listdir(self.root_dir)
                if os.path.isdir(self.root_dir + '/' + sub)]
            for sub in subs:
                os.makedirs(target_folder + '/' + sub)
        
        do_lrflip = np.random.rand(self.length)
        _loop = tqdm(range(self.length), ncols=80)
        for i in _loop:
            name = self.imagename[i]
            img = imread(self.root_dir + name)
            img = img.astype(np.float) / 255.
            img = self._rand_crop_and_resize(img, i)
            if do_lrflip[i] > 0.5:
                img = self._lrflip(img, i)
            img = self._add_noise(img, sigma_noise)
            '''
            self._visualize(
                '_test_washed/result_{}.png'.format(i),
                img, self.gt2d[i, :, :2]
            )
            '''
            #imsave(target_folder + name, img)
            imwrite(target_folder + name, (img*255).astype(np.uint8))
        
        keep_list = ['gt2d', 'gt3d', 'imagename', 'pose', 'shape']
        f = open(target_folder + '/' + annotation_pickle, 'wb')
        keep_dict = {k: getattr(self, k) for k in keep_list}
        pickle.dump(keep_dict, f)
        f.close()
        
if __name__ == '__main__':
    np.random.seed(9608)
    root_dir = None  # Change this to your human36m dataset path
    datawasher = DataWasher(root_dir=root_dir)
    datawasher._strip_non_square_images()
    datawasher.data_augmentation()
