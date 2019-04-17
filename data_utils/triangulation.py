from smpl_torch_batch import SMPLModel
import numpy as np
import torch

def import_verts(obj_file):
    with open(obj_file, 'r') as f:
        lines = [
            line for line in f
            if line.startswith('v ')
        ]
        
    verts = np.zeros((6890, 3), dtype=np.float)
    for (i, line) in enumerate(lines):
        ls = line.split(' ')
        verts[i,0] = float(ls[1])
        verts[i,1] = float(ls[2])
        verts[i,2] = float(ls[3])
    return verts

def triangulation(quad_obj, out_obj, trifaces):
    with open(quad_obj, 'r') as f:
        all_lines = [line for line in f]
        
    is_face = [line.startswith('f ') for line in all_lines]
    fid = is_face.index(True)
    other_lines = all_lines[:fid]
    face_lines = all_lines[fid:]
        
    # convert each quad into two triangles
    # that matches two corresponding lines in trifaces
    # I assume that SMPL faces are converted this way from fbx
    triface_pointer = 0
    tri_face_lines = []
    for line in face_lines:
        ls = [item.replace('\n', '') for item in line.split(' ')[1:]] # weird
        if len(ls) == 4: # a quad
            vs = {int(triplet.split('/')[0]):triplet for triplet in ls}
            triangle_1 = trifaces[triface_pointer]
            # note that in SMPLModel objects vert index starts with 0 rather than 1
            tri_line_1 = ' '.join(['f', vs[triangle_1[0]+1], vs[triangle_1[1]+1], vs[triangle_1[2]+1]])
            tri_face_lines.append(tri_line_1 + '\n')
            
            triangle_2 = trifaces[triface_pointer+1]
            tri_line_2 = ' '.join(['f', vs[triangle_2[0]+1], vs[triangle_2[1]+1], vs[triangle_2[2]+1]])
            tri_face_lines.append(tri_line_2 + '\n')
            
            triface_pointer += 2
        elif len(ls) == 3: # a triangle
            tri_face_lines.append(line+'\n')
            triface_pointer += 1
        else:
            print('wtf?')
    
    print(len(tri_face_lines), triface_pointer)
    assert(len(tri_face_lines) == trifaces.shape[0])
    with open(out_obj, 'w') as f:
        f.writelines(other_lines + tri_face_lines)
       
if __name__ == '__main__':
    device=torch.device('cuda')
    data_type=torch.float32
    pose_size = 72
    beta_size = 10

    model = SMPLModel(
                device=device,
                model_path = './model_lsp.pkl',
                data_type=data_type,
                simplify=True
            )
            
    quad_obj = 'untitled.obj'
    out_obj = 'smpl_fbx_template.obj'
    fbx_obj_verts = import_verts(quad_obj)
    #model.write_obj(fbx_obj_verts, 'hybrid_fbx_verts_SMPL_face.obj')
    triangulation(quad_obj, out_obj, model.faces)
