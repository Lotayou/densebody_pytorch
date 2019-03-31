import numpy as np
from skimage.io import imread, imsave
from skimage.draw import polygon_perimeter as pope
import pickle

class UV_Texture_Parser():
    def __init__(self, load_pickle=None):
        self.vt_num = -1
        self.face_num = -1
        self.obj_name = None
        self.vts = None
        self.faces = None
        self.vt_to_v = None  # texture vertices to 3D vertices index
        
        if not load_pickle is None:
            f = open(load_pickle, 'rb')
            tmp = pickle.load(f)
            f.close()
            self.vts = tmp['vts']
            self.faces = tmp['faces']
            self.vt_to_v = tmp['vt_to_v']
        
    def parse_obj(self, obj_file):
        with open(obj_file, 'r') as fin:
            lines = [l 
                for l in fin.readlines()
                if len(l.split()) > 0
                and not l.startswith('#')
            ]
        
        self.obj_name = obj_file
        
        # load vertices
        vertices = []
        for line in lines:
            lsp = line.split()
            if lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                vertices.append((1-v, u))
                
        self.vts = np.vstack(vertices).astype(np.float32)
        
        # load face_data
        # assume all lines are like:
        # f v1/vt1/vn1 v1/vt1/vn1 v1/vt1/vn1
        faces = []
        self.vt_to_v = {}
        for line in lines:
            if line.split()[0] == 'f':
                vs = line.split()
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                faces.append((vt0, vt1, vt2))
                self.vt_to_v[vt0] = v0
                self.vt_to_v[vt1] = v1
                self.vt_to_v[vt2] = v2
                
        self.faces = np.vstack(faces)
        
    def save_UV_data(self, out_pickle_name='SMPL_UV_map.pickle'):
        if self.faces is None:
            print('Cyka Blyat: Load an obj file first!')
        
        tmp_dict = {
            'vts': self.vts,
            'faces': self.faces,
            'vt_to_v': self.vt_to_v
        }
        with open(out_pickle_name, 'wb') as w:
            pickle.dump(tmp_dict, w)
        
    def render_UV_map(self, image_name, size=1024):
        # Just draw each edge twice, no biggie
        if self.faces is None:
            print('Cyka Blyat: Load an obj file first!')
        
        faces = (self.vts[self.faces] * size).astype(np.int32)
        img = np.zeros((size, size), dtype=np.uint8)
        for f in faces:
            rr, cc = pope(f[:,0], f[:,1], shape=(size,size))
            img[rr,cc] = 255
            
        imsave(image_name, img)
            
if __name__ == '__main__':
#    parser = UV_Texture_Parser(load_pickle='SMPL_UV_map.pickle')
    parser = UV_Texture_Parser()
    parser.parse_obj('SMPL_template_UV_map.obj')
    parser.render_UV_map('SMPL_UV_map.png')
    parser.save_UV_data()
