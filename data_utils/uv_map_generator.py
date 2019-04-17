import numpy as np
import pickle
import torch
from torch.nn import Module
import os
import shutil
from sys import platform
from skimage.io import imread, imsave
from skimage.draw import circle
from skimage.draw import polygon_perimeter as pope

from time import time
from tqdm import tqdm
from numpy.linalg import solve
from scipy.interpolate import RectBivariateSpline as RBS

'''
    UV_Map_Generator: preparing UV position map labels 
    and resample 3D vertex coords from rendered UV maps.
'''
class UV_Map_Generator():
    def __init__(self, UV_height, UV_width=-1, 
        UV_pickle='radvani_template.pickle'):
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        
        ### Load UV texcoords and face mapping info
        if not os.path.isfile(UV_pickle):
            self._parse_obj(
                UV_pickle.replace('pickle','obj'), UV_pickle
            )
        else:
            with open(UV_pickle, 'rb') as f:
                tmp = pickle.load(f)
            for k in tmp.keys():
                setattr(self, k, tmp[k])
        
        ### Load (or calcluate) barycentric info
        self.bc_pickle = 'barycentric_h{:04d}_w{:04d}_{}'\
            .format(self.h, self.w, UV_pickle)
        if os.path.isfile(self.bc_pickle):
            print('Find cached pickle file...')
            with open(self.bc_pickle, 'rb') as rf:
                bary_info = pickle.load(rf)
                self.bary_id = bary_info['face_id']
                self.bary_weights = bary_info['bary_weights']
                self.edge_dict = bary_info['edge_dict']
                
        else:
            print('Bary info cache not found, start calculating...' 
                + '(This could take a few minutes)')
            self.bary_id, self.bary_weights, self.edge_dict = \
                self._calc_bary_info(self.h, self.w, self.vt_faces.shape[0])
    
    #####################
    # Private Functions #
    #####################
    def _parse_obj(self, obj_file, cache_file):
        with open(obj_file, 'r') as fin:
            lines = [l 
                for l in fin.readlines()
                if len(l.split()) > 0
                and not l.startswith('#')
            ]
        
        # Load all vertices (v) and texcoords (vt)
        vertices = []
        texcoords = []
        
        for line in lines:
            lsp = line.split()
            if lsp[0] == 'v':
                x = float(lsp[1])
                y = float(lsp[2])
                z = float(lsp[3])
                vertices.append((x, y, z))
            elif lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                texcoords.append((1 - v, u))
                
        # Stack these into an array
        self.vertices = np.vstack(vertices).astype(np.float32)
        self.texcoords = np.vstack(texcoords).astype(np.float32)
        
        # Load face data. All lines are of the form:
        # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        #
        # Store the texcoord faces and a mapping from texcoord faces
        # to vertex faces
        vt_faces = []
        self.vt_to_v = {}
        self.v_to_vt = [None] * self.vertices.shape[0]
        for i in range(self.vertices.shape[0]):
            self.v_to_vt[i] = set()
        
        for line in lines:
            vs = line.split()
            if vs[0] == 'f':
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                vt_faces.append((vt0, vt1, vt2))
                self.vt_to_v[vt0] = v0
                self.vt_to_v[vt1] = v1
                self.vt_to_v[vt2] = v2
                self.v_to_vt[v0].add(vt0)
                self.v_to_vt[v1].add(vt1)
                self.v_to_vt[v2].add(vt2)
                
        self.vt_faces = np.vstack(vt_faces)
        tmp_dict = {
            'vertices': self.vertices,
            'texcoords': self.texcoords,
            'vt_faces': self.vt_faces,
            'vt_to_v': self.vt_to_v,
            'v_to_vt': self.v_to_vt
        }
        with open(cache_file, 'wb') as w:
            pickle.dump(tmp_dict, w)
            
    '''
    _calc_bary_info: for a given uv vertice position,
    return the berycentric information for all pixels
    
    Parameters:
    ------------------------------------------------------------
    h, w: image size
    faces: [F * 3], triangle pieces represented with UV vertices.
    uvs: [N * 2] UV coordinates on texture map, scaled with h & w
    
    Output: 
    ------------------------------------------------------------
    bary_dict: A dictionary containing three items, saved as pickle.
    @ face_id: [H*W] int tensor where f[i,j] represents
        which face the pixel [i,j] is in
        if [i,j] doesn't belong to any face, f[i,j] = -1
    @ bary_weights: [H*W*3] float tensor of barycentric coordinates 
        if f[i,j] == -1, then w[i,j] = [0,0,0]
    @ edge_dict: {(u,v):n} dict where (u,v) indicates the pixel to 
        dilate, with n being non-zero neighbors within 8-neighborhood
        
        
    Algorithm:
    ------------------------------------------------------------
    The barycentric coordinates are obtained by 
    solving the following linear equation:
    
            [ x1 x2 x3 ][w1]   [x]
            [ y1 y2 y3 ][w2] = [y]
            [ 1  1  1  ][w3]   [1]
    
    Note: This algorithm is not the fastest but can be batchlized.
    It could take 8~10 minutes on a regular PC for 300*300 maps. 
    Luckily, for each experiment the bary_info only need to be 
    computed once, so I just stick to the current implementation.
    '''
    
    def _calc_bary_info(self, h, w, F):
        s = time()
        face_id = np.zeros((h, w), dtype=np.int)
        bary_weights = np.zeros((h, w, 3), dtype=np.float32)
        
        uvs = self.texcoords * np.array([[self.h - 1, self.w - 1]])
        grids = np.ones((F, 3), dtype=np.float32)
        anchors = np.concatenate((
            uvs[self.vt_faces].transpose(0,2,1),
            np.ones((F, 1, 3), dtype=uvs.dtype)
        ), axis=1) # [F * 3 * 3]
        
        _loop = tqdm(np.arange(h*w), ncols=80)
        for i in _loop:
            r = i // w
            c = i % w
            grids[:, 0] = r
            grids[:, 1] = c
            
            weights = solve(anchors, grids) # not enough accuracy?
            inside = np.logical_and.reduce(weights.T > 1e-10)
            index = np.where(inside == True)[0]
            
            if 0 == index.size:
                face_id[r,c] = -1  # just assign random id with all zero weights.
            #elif index.size > 1:
            #    print('bad %d' %i)
            else:
                face_id[r,c] = index[0]
                bary_weights[r,c] = weights[index[0]]

        # calculate counter pixels for UV_map dilation
        _mask = np.where(face_id == -1, 0, 1)
        edge_dict = {}
        _loop = _loop = tqdm(np.arange((h-2)*(w-2)), ncols=80)
        for l in _loop:
            i = l // (w-2) + 1 
            j = l % (w-2) + 1
            _neighbor = np.array([
                _mask[i-1, j], _mask[i-1, j+1],
                _mask[i, j+1], _mask[i+1, j+1],
                _mask[i+1, j], _mask[i+1, j-1],
                _mask[i, j-1], _mask[i-1, j-1],
            ])
            
            if _mask[i,j] == 0 and _neighbor.min() != _neighbor.max():
                edge_dict[(i, j)] = np.count_nonzero(_neighbor)
                    
            
        print('Calculating finished. Time elapsed: {}s'.format(time()-s))
        
        bary_dict = {
            'face_id': face_id,
            'bary_weights': bary_weights,
            'edge_dict': edge_dict
        }
        
        with open(self.bc_pickle, 'wb') as wf:
            pickle.dump(bary_dict, wf)
            
        return face_id, bary_weights, edge_dict
        
    '''
        _dilate: perform dilate_like operation for initial
        UV map, to avoid out-of-region error when resampling
    '''
    def _dilate(self, UV_map, pixels=1):
        _UV_map = UV_map.copy()
        for k, v in self.edge_dict.items():
            i, j = k[0], k[1]
            _UV_map[i, j] = np.sum(np.array([
                UV_map[i-1, j], UV_map[i-1, j+1],
                UV_map[i, j+1], UV_map[i+1, j+1],
                UV_map[i+1, j], UV_map[i+1, j-1],
                UV_map[i, j-1], UV_map[i-1, j-1],
            ]), axis=0) / v
    
        return _UV_map
        
    ####################
    # Render Functions #
    ####################
    
    '''
        Render a point cloud to an image of [self.h, self.w] size. This point cloud
        approximates what the UV position map will look like: for each texture
        coordinate, we render its corresponding _vertex_ by setting the 
        vertex's (normalized) XYZ to the RGB of the pixel corresponding to the
        texture coordinate.
        
        verts: [N * 3] vertex coordinates
        rgbs: [N * 3] RGB values in [0,1] float.
            If vertex color is not defined, 
            the normalized XYZ coordinates will be used
    '''
    def render_point_cloud(self, img_name=None, verts=None, rgbs=None, eps=1e-8):
        if verts is None:
            verts = self.vertices
        if rgbs is None:
            #print('Warning: rgb not specified, use normalized 3d coords instead...')
            v_min = np.amin(verts, axis=0, keepdims=True)
            v_max = np.amax(verts, axis=0, keepdims=True)
            rgbs = (verts - v_min) / np.maximum(eps, v_max - v_min)
        
        vt_id = [self.vt_to_v[i] for i in range(self.texcoords.shape[0])]
        img = np.zeros((self.h, self.w, 3), dtype=rgbs.dtype)
        uvs = (self.texcoords * np.array([[self.h - 1, self.w - 1]])).astype(np.int)
        
        img[uvs[:, 0], uvs[:, 1]] = rgbs[vt_id]
        
        if img_name is not None:
            imsave(img_name, img)
            
        return img, verts, rgbs
        
    def render_UV_atlas(self, image_name, size=1024):
        if self.vt_faces is None:
            print('Cyka Blyat: Load an obj file first!')
        
        faces = (self.texcoords[self.vt_faces] * size).astype(np.int32)
        img = np.zeros((size, size), dtype=np.uint8)
        for f in faces:
            rr, cc = pope(f[:,0], f[:,1], shape=(size, size))
            img[rr, cc] = 255
            
        imsave(image_name, img)
    
    def write_ply(self, ply_name, verts, rgbs=None, eps=1e-8):
        if rgbs is None:
            #print('Warning: rgb not specified, use normalized 3d coords instead...')
            v_min = np.amin(verts, axis=0, keepdims=True)
            v_max = np.amax(verts, axis=0, keepdims=True)
            rgbs = (verts - v_min) / np.maximum(eps, v_max - v_min)
        if rgbs.max() < 1.001:
            rgbs = (rgbs * 255.).astype(np.uint8)
        
        with open(ply_name, 'w') as f:
            # headers
            f.writelines([
                'ply\n'
                'format ascii 1.0\n',
                'element vertex {}\n'.format(verts.shape[0]),
                'property float x\n',
                'property float y\n',
                'property float z\n',
                'property uchar red\n',
                'property uchar green\n',
                'property uchar blue\n',
                'end_header\n',
                ]
            )
            
            for i in range(verts.shape[0]):
                str = '{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n'\
                    .format(verts[i,0], verts[i,1], verts[i,2],
                        rgbs[i,0], rgbs[i,1], rgbs[i,2])
                f.write(str)
                
        return verts, rgbs
    
    #####################
    # UV Map Generation #
    #####################
    '''
    UV_interp: barycentric interpolation from given
    rgb values at non-integer UV vertices.
    
    Parameters:
    ------------------------------------------------------------
    rgbs: [N * 3] rgb colors at given uv vetices.
    
    Output: 
    ------------------------------------------------------------
    UV_map: colored texture map with the same size as im
    '''    
    def UV_interp(self, rgbs):
        face_num = self.vt_faces.shape[0]
        vt_num = self.texcoords.shape[0]
        assert(vt_num == rgbs.shape[0])
        
        uvs = self.texcoords * np.array([[self.h - 1, self.w - 1]])
        
        #print(np.max(rgbs), np.min(rgbs))
        triangle_rgbs = rgbs[self.vt_faces][self.bary_id]
        bw = self.bary_weights[:,:,np.newaxis,:]
        #print(triangle_rgbs.shape, bw.shape)
        im = np.matmul(bw, triangle_rgbs).squeeze(axis=2)
        
        '''
        for i in range(height):
            for j in range(width):
                t0 = faces[fid[i,j]][0]
                t1 = faces[fid[i,j]][1]
                t2 = faces[fid[i,j]][2]
                im[i,j] = (w[i,j,0] * rgbs[t0] + w[i,j,1] * rgbs[t1] + w[i,j,2] * rgbs[t2])
        '''
        
        #print(im.shape, np.max(im), np.min(im))
        im = np.minimum(np.maximum(im, 0.), 1.)
        return im
        
    '''
    get_UV_map: create UV position map from aligned mesh coordinates
    Parameters:
    ------------------------------------------------------------
    verts: [V * 3], aligned mesh coordinates.
    
    Output: 
    ------------------------------------------------------------
    UV_map: [H * W * 3] Interpolated UV map.
    colored_verts: [H * W * 3] Scatter plot of colorized UV vertices
    '''
    def get_UV_map(self, verts):
        # normalize all to [0,1]
        _min = np.amin(verts, axis=0, keepdims=True)
        _max = np.amax(verts, axis=0, keepdims=True)
        verts = (verts - _min) / (_max - _min)
        verts_backup = verts.copy()
        
        vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])
        rgbs = verts[vt_to_v_index]
        
        #return self.UV_interp(rgbs), verts_backup
        return self._dilate(self.UV_interp(rgbs)), verts_backup
    
    '''
        TODO: make it torch.
    '''
    def resample(self, UV_map):
        h, w, c = UV_map.shape
        vts = np.floor(self.texcoords * np.array([[self.h - 1, self.w - 1]])).astype(np.int)
        vt_3d = [None] * vts.shape[0]
        
        for i in range(vts.shape[0]):
            
            coords = [
                (vts[i, 0], vts[i, 1]),
                (vts[i, 0], vts[i, 1]+1),
                (vts[i, 0]+1, vts[i, 1]),
                (vts[i, 0]+1, vts[i, 1]+1),
            ]
            for coord in coords:
                if UV_map[coord[0], coord[1]].max() > 0:
                    vt_3d[i] = UV_map[coord[0], coord[1]]
                    
        vt_3d = np.stack(vt_3d)
        # convert vt back to v (requires v_to_vt index)
        cyka_v_3d = [None] * len(self.v_to_vt)
        for i in range(len(self.v_to_vt)):
            cyka_v_3d[i] = np.mean(vt_3d[list(self.v_to_vt[i])], axis=0)
        
        cyka_v_3d = np.array(cyka_v_3d)
        return cyka_v_3d

        
if __name__ == '__main__':
    # test render module
    # change this to the same as in train.py opt.uv_prefix
    file_prefix = 'radvani_template'
    #file_prefix = 'vbml_close_template'
    #file_prefix = 'vbml_spaced_template'
    generator = UV_Map_Generator(
        UV_height=256,
        UV_pickle=file_prefix+'.pickle'
    )
    test_folder = '_test_radvani'
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
        
    generator.render_UV_atlas('{}/{}_atlas.png'.format(test_folder, file_prefix))
    img, verts, rgbs = generator.render_point_cloud('{}/{}.png'.format(test_folder, file_prefix))
    verts, rgbs = generator.write_ply('{}/{}.ply'.format(test_folder, file_prefix), verts, rgbs)
    uv, _ = generator.get_UV_map(verts)
    uv = uv.max(axis=2)
    print(uv.shape)
    binary_mask = np.where(uv > 0, 1., 0.)
    binary_mask = (binary_mask * 255).astype(np.uint8)
    imsave('{}_UV_mask.png'.format(file_prefix), binary_mask)
