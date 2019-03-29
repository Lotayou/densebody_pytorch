import numpy as np
from skimage.io import imread, imsave
from skimage.draw import polygon_perimeter as pope

class UV_Texture_Parser():
    def __init__(self):
        self.vt_num = -1
        self.face_num = -1
        self.obj_name = None
        self.vts = None
        self.faces = None
        
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
        for line in lines:
            if line.split()[0] == 'f':
                vs = line.split()
                v0 = vs[1].split('/')[1]
                v1 = vs[2].split('/')[1]
                v2 = vs[3].split('/')[1]
                faces.append((v0, v1, v2))
                
        self.faces = np.vstack(faces).astype(np.int32) - 1
        
    def save_UV_data(self, out_pickle_name):
        pass
        
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
    parser = UV_Texture_Parser()
    parser.parse_obj('SMPL_template_UV_map.obj')
    parser.render_UV_map('SMPL_UV_map.png')
