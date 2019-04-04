import numpy as np
import torch
import pickle
from skimage.io import imsave
from time import time
import os
from tqdm import tqdm
from numpy.linalg import solve
from scipy.interpolate import RectBivariateSpline as RBS
    
def get_barycentric_info(h, w, uvs, faces):
    bc_pickle = 'barycentric_h{:04d}_w{:04d}.pickle'.format(h, w)
    if os.path.isfile(bc_pickle):
        print('Find cached pickle file...')
        with open(bc_pickle, 'rb') as rf:
            bary_info = pickle.load(rf)
        return bary_info['face_id'], bary_info['bary_coords']
    else:
        print('Bary info cache not found, start calculating...' 
            + '(This could take a few minutes)')
        s = time()
        F = faces.shape[0]
        face_id = np.zeros((h, w), dtype=np.int)
        bary_weights = np.zeros((h, w, 3), dtype=np.float32)
        grids = np.ones((F, 3), dtype=np.float32)
        anchors = np.concatenate((
            uvs[faces].transpose(0,2,1),
            np.ones((F, 1, 3), dtype=uvs.dtype)
        ), axis=1) # [F * 3 * 3]
        
        _loop = tqdm(np.arange(h*w), ncols=80)
        for i in _loop:
        # for i in range(h*w):
            r = i // w
            c = i % w
            grids[:, 0] = r
            grids[:, 1] = c
            
            weights = solve(anchors, grids) # not enough accuracy
            #if i == 50:
                #print(np.matmul(anchors, weights[:,:,np.newaxis]))
            inside = np.logical_and.reduce(weights.T > 1e-10)
            index = np.where(inside == True)[0]
            
            if 0 == index.size:
                face_id[r,c] = 0  # just assign random id with all zero weights.
            elif index.size > 1:
                print('bad %d' %i)
            else:
                face_id[r,c] = index[0]
                bary_weights[r,c] = weights[index[0]]
                
        print('Calculating finished. Time elapsed: {}s'.format(time()-s))
        
        bary_dict = {
            'face_id': face_id,
            'bary_coords': bary_weights
        }
        
        with open(bc_pickle, 'wb') as wf:
            pickle.dump(bary_dict, wf)
            
        return face_id, bary_weights
    
'''
    UV_interp: barycentric interpolation from given
    rgb values at non-integer UV vertices.
    
    Parameters:
    ------------------------------------------------------------
    im: [H * W * 3] empty canvas with predefined height and width
    faces: [F * 3], triangle pieces represented with UV vertices.
    uvs: [N * 2] UV coordinates on texture map
    rgbs: [N * 3] rgb colors at given uv vetices.
    
    Output: 
    ------------------------------------------------------------
    UV_map: colored texture map with the same size as im
    
'''   
def UV_interp(im, faces, uvs, rgbs):
    height = im.shape[0]
    width = im.shape[1]
    face_num = faces.shape[0]
    vt_num = uvs.shape[0]
    assert(vt_num == rgbs.shape[0])
    
    uvs *= np.array([[height - 1, width - 1]])
    
    # find barycentric coordinates, this is time 
    # consuming but also only need to be computed once
    # (If the height and width of the image is fixed)
    fid, w = get_barycentric_info(height, width, uvs, faces)
    
    #print(np.max(rgbs), np.min(rgbs))
    
    #print(rgbs.shape, faces.shape, fid.shape)
    
    #triangle_rgbs = rgbs[faces][fid]
    #print(triangle_rgbs.shape)
    # w = w[:,:,:,np.newaxis]
    #print(triangle_rgbs.shape, w.shape)
    
    #im = np.matmul(triangle_rgbs, w).squeeze(axis=3)
    for i in range(height):
        for j in range(width):
            t0 = faces[fid[i,j]][0]
            t1 = faces[fid[i,j]][1]
            t2 = faces[fid[i,j]][2]
            im[i,j] = (w[i,j,0] * rgbs[t0] + w[i,j,1] * rgbs[t1] + w[i,j,2] * rgbs[t2])
        
    #print(im.shape, np.max(im), np.min(im))
    im = np.minimum(np.maximum(im, 0.), 1.)
    return im
    
    
'''
    get_UV_position_map: create UV position map from aligned mesh coordinates
    Parameters:
    ------------------------------------------------------------
    verts: [V * 3], aligned mesh coordinates.
    height, width: result UV map size.
    UV_data_pickle: pickle file that specifies UV texture map coordinates.
    
    Output: 
    ------------------------------------------------------------
    UV_map: [H * W * 3] Interpolated UV map.
    colored_verts: [H * W * 3] Scatter plot of colorized UV vertices
    
'''
def get_UV_position_map(verts, height, width=0, 
        UV_data_pickle='SMPL_template_UV_map.pickle'):
        
    if width == 0:
        width = height
        
    # normalize all to [0,1]
    _min = np.amin(verts, axis=0, keepdims=True)
    _max = np.amax(verts, axis=0, keepdims=True)
    
    scale_vector = _max - _min
    #scale_vector[:, 0] = height
    #scale_vector[:, 1] = width
    verts = (verts - _min) / scale_vector
    
    # if y > 0.5 verts blue
    # else color red
    '''
    ones_vec = np.ones(verts.shape[0], dtype=verts.dtype)
    zeros_vec = np.zeros(verts.shape[0], dtype=verts.dtype)
    verts = np.stack((
        np.where(verts[:,1] > 0.5, ones_vec, zeros_vec),
        zeros_vec,
        np.where(verts[:,1] > 0.5, zeros_vec, ones_vec),
        ),axis=1
    )
    '''
    rgbs_backup = verts.copy()
    
    # load necessary components
    f = open(UV_data_pickle, 'rb')
    tmp = pickle.load(f)
    f.close()
    _vts = tmp['vts']
    _faces = tmp['faces']
    _vt_to_v = tmp['vt_to_v']
    del tmp
    
    _tmp_UV_map = np.zeros((height, width, 3), dtype=verts.dtype)
    
    rgbs = np.zeros((_vts.shape[0], 3), dtype=verts.dtype)
    for i in range(_vts.shape[0]):
        rgbs[i] = verts[_vt_to_v[i]]
    
    UV_map = UV_interp(_tmp_UV_map, _faces, _vts, rgbs)  # weird, _vts value should not change...
    
    # basic points color visualization
    _vts = _vts.astype(np.int)
    UV_scatter = np.zeros((height, width, 3), dtype=verts.dtype)
    UV_scatter[_vts[:, 0], _vts[:, 1]] = rgbs
    
    return UV_map, UV_scatter, rgbs_backup
    

    
if __name__ == '__main__':
    get_UV_position_map(None, 300)

    