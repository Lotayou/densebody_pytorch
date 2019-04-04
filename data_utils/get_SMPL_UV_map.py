import numpy as np
from skimage.io import imread, imsave
from skimage.draw import polygon_perimeter as pope
import pickle

class UV_Texture_Parser():
    def __init__(self, load_pickle=None):        
        self.vertices = None
        self.texcoords = None
        self.vt_faces = None
        self.vt_to_v = None
        
        if not load_pickle is None:
            f = open(load_pickle, 'rb')
            tmp = pickle.load(f)
            f.close()
            
            self.vertices = tmp['vertices']
            self.texcoords = tmp['texcoords']
            self.vt_faces = tmp['vt_faces']
            self.vt_to_v = tmp['vt_to_v']
        
    def parse_obj(self, obj_file):
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

        for line in lines:
            if line.split()[0] == 'f':
                vs = line.split()
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                
                vt_faces.append((vt0, vt1, vt2))
                self.vt_to_v[(vt0, vt1, vt2)] = (v0, v1, v2)

        self.vt_faces = np.vstack(vt_faces)
        
    def save_UV_data(self, out_pickle_name='SMPL_UV_map.pickle'):
        if self.vt_faces is None:
            print('Cyka Blyat: Load an obj file first!')

        tmp_dict = {
            'vertices': self.vertices,
            'texcoords': self.texcoords,
            'vt_faces': self.vt_faces,
            'vt_to_v': self.vt_to_v
        }
        with open(out_pickle_name, 'wb') as w:
            pickle.dump(tmp_dict, w)
        
    def render_UV_map(self, image_name, size=1024):
        # Just draw each edge twice, no biggie
        if self.vt_faces is None:
            print('Cyka Blyat: Load an obj file first!')
        
        faces = (self.texcoords[self.vt_faces] * size).astype(np.int32)
        img = np.zeros((size, size), dtype=np.uint8)
        for f in faces:
            rr, cc = pope(f[:,0], f[:,1], shape=(size, size))
            img[rr, cc] = 255
            
        imsave(image_name, img)

    def render_point_cloud(self, image_name, size=300):
        """
        Render a point cloud to an image of the given size. This point cloud
        approximates what the UV position map will look like: for each texture
        coordinate, we render its corresponding _vertex_ by setting the 
        vertex's (normalized) XYZ to the RGB of the pixel corresponding to the
        texture coordinate.
        """        
        x = []
        y = []
        rgb = []
        
        v_min = np.amin(self.vertices, axis=0, keepdims=True)
        v_max = np.amax(self.vertices, axis=0, keepdims=True)
        print("Mininum vertex (for normalization) {}".format(v_min))
        print("Maximum vertex (for normalization) {}".format(v_max))

        self.vertices -= v_min
        self.vertices = self.vertices / (v_max - v_min)
        
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for face in self.vt_faces:
            vt0, vt1, vt2 = face
            v0, v1, v2 = self.vt_to_v[(vt0, vt1, vt2)]
            
            vertices = (self.vertices[v0], self.vertices[v1], self.vertices[v2])
            texcoords = (self.texcoords[vt0], self.texcoords[vt1], self.texcoords[vt2])
        
            for i in range(len(texcoords)):
                texcoord = texcoords[i] * size
                vertex = np.array(vertices[i]) * 255
                img[int(texcoord[0]), int(texcoord[1])] = vertex
                
        imsave(image_name, img)

if __name__ == '__main__':
    # To generate the UV map from an OBJ
    parser = UV_Texture_Parser()

    file_prefix = "SMPL_template_UV_map"
    parser.parse_obj("{}.obj".format(file_prefix))
    parser.render_UV_map("{}.png".format(file_prefix))
    parser.save_UV_data("{}.pickle".format(file_prefix))
    parser.render_point_cloud("{}_cloud.png".format(file_prefix))

    


