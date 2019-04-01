import numpy as np
import torch
import pickle
from skimage.io import imsave
from time import time
import os
from tqdm import tqdm
from numpy.linalg import solve

#import neural_renderer.cuda.create_texture_image as create_texture_image_cuda

'''
    im2texture: Get RGB value at each mesh vertex from the given image
    Parameters:
    ------------------------------------------------------------
    im: [N * H * W * 3] numpy array
    verts: [N * 6890 * 3], aligned mesh coordinates.
    
    Output: 
    ------------------------------------------------------------
    texture_tensor: [N * 6890 * 3]: rgb values at each vertex
    
    Algorithm:
    ------------------------------------------------------------
    @ Bilinear Interpolation
'''
def im2texture(im, verts):
    texture_tensor = torch.zeros_like(verts)
    return texture_tensor
    
    
'''
    get_barycentric_info: for a given uv vertice position,
    return the berycentric information for all pixels
    
    Parameters:
    ------------------------------------------------------------
    h, w: image size
    faces: [F * 3], triangle pieces represented with UV vertices.
    uvs: [N * 2] UV coordinates on texture map, scaled with h & w
    
    Output: 
    ------------------------------------------------------------
    bary_info: A dictionary containing two items:
    @ index: [H*W] int tensor where f[i,j] represents
        which face the pixel [i,j] is in
        if [i,j] doesn't belong to any face, f[i,j] = -1
    @ coords: [H*W*3] float tensor of barycentric coordinates 
        if f[i,j] == -1, then coords[i,j] = [0,0,0]
        
    Note: This function could be time consuming, luckily, for our
    experiments the UV_texture only need to be compute once, so I
    just stick to CPU implementation.
'''
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
        
        # 20190401 direct calculation impossible, too memory heavy
        
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
            r = i // w
            c = i % w
            grids[:, 0] = r
            grids[:, 1] = c
            
            weights = solve(anchors, grids).T
            inside = np.logical_and.reduce(weights > 1e-10)
            index = np.where(inside == True)[0]
            
            if 0 == index.size:
                face_id[r,c] = 0  # just assign random id with all zero weights.
            else:
                face_id[r,c] = index[0]
                bary_weights[r,c] = weights[:, index[0]]
                
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
    
    triangle_rgbs = rgbs[faces][fid]
    w = w[:,:,:,np.newaxis]
    #print(triangle_rgbs.shape, w.shape)
    #print(triangle_rgbs[0:5,0:5], w[0:5,0:5]) 
    
    im = np.sum(triangle_rgbs * w, axis=2)
    #print(im.shape, np.max(im), np.min(im))
    im = np.minimum(np.maximum(im*2.-1., -1.), 1.)
    return im
'''
    get_UV_position_map: create UV position map from aligned mesh coordinates
    Parameters:
    ------------------------------------------------------------
    verts: [6890 * 3], aligned mesh coordinates.
    height, width: result UV map size.
    UV_data_pickle: pickle file that specifies UV texture map coordinates.
    
    Output: 
    ------------------------------------------------------------
    UV_map: [H * W * 3] float32 ndarray.
    
'''
def get_UV_position_map(verts, height, width=0, 
        UV_data_pickle='SMPL_UV_map.pickle'):
        
    if width == 0:
        width = height
        
    # normalize all to [0,1]
    _min = np.amin(verts, axis=0, keepdims=True)
    _max = np.amax(verts, axis=0, keepdims=True)
    verts -= _min
    verts = verts / (_max - _min)
    #verts = (verts - .5) * 2.
    
    # load necessary components
    f = open(UV_data_pickle, 'rb')
    tmp = pickle.load(f)
    f.close()
    _vts = tmp['vts']
    _faces = tmp['faces']
    _vt_to_v = tmp['vt_to_v']
    del tmp
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    vid = [_vt_to_v[i] for i in range(_vts.shape[0])]
    img = UV_interp(img, _faces, _vts, verts[vid])
    
    return img
    
    
if __name__ == '__main__':
    get_UV_position_map(None, 300)

    