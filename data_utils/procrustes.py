import numpy as np
import torch
from torch.nn import Module
import os
from sys import platform

'''
    procrustes_3d_to_2d: 
        Align 3D input keypoints to 2D/3D ground truth via Procrustes analysis
        
    Parameters:
    ------------------------------------------------------------
    J3d: [N * J * 3]: batch 3D input 
    gt3d: groung truth 3D input
    gt2d: groung truth 2D input
    
    Output:
    ------------------------------------------------------------
    A transformation function that directly takes input N * None * 3 points
    and align it with 2D annotations.
    
    Algorithm:
    ------------------------------------------------------------
    @ Using ground truth 3D to determine rotation
    @ Using ground truth 2D to determine translation and scaling
'''    
def map_3d_to_2d(J3d, gt2d, gt3d):
    batch_size = J3d.shape[0]
    
    # Part 1: Align J3d with gt3d
    
    ### i. centering: must
    G3d = gt3d.clone()
    cent_J = torch.mean(J3d, dim=1, keepdim=True)
    J3d -= cent_J
    cent_G = torch.mean(G3d, dim=1, keepdim=True)
    G3d -= cent_G
    
    ### ii. scaling: not necessary here.
    
    ### iii. rotation
    M = torch.bmm(G3d.transpose(1,2), J3d) # [N, 3, 3]
    if platform == 'linux':
        U, D, V = batch_svd(M)
        R = torch.bmm(V, U.transpose(1,2))
    else:
        R = [None] * batch_size
        for i in range(batch_size):
            U, D, V = torch.svd(M[i])
            R[i] = torch.mm(V, U.transpose(0,1)) # transpose
        R = torch.stack(R, dim=0)
        
    reg3d = torch.bmm(J3d, R)
    ### eval stage I: rel error < 5%
    '''
    print(reg3d - G3d)
    test_case = range(10)
    for i in test_case:
        #np.savetxt('_test_cache/2d_gt_{}.xyz'.format(i), gt2d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/3d_in_{}.xyz'.format(i), J3d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/3d_reg_{}.xyz'.format(i), reg3d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/3d_gt_{}.xyz'.format(i), G3d[i].detach().cpu().numpy(), delimiter=' ')
    '''
    
    # Part II: Align reg3d with gt2d
    G2d = gt2d.clone()
    reg2d = reg3d[:, :, :2]
    ### i. translation
    cent2d = torch.mean(G2d, dim=1, keepdim=True)
    G2d -= cent2d
    cent3d = torch.cat((cent2d,
        torch.zeros((batch_size, 1, 1), dtype=cent2d.dtype, device=cent2d.device)
    ), dim=2)
    
    ### ii. scaling
    G2d = G2d.view(batch_size, -1)
    r2d = reg2d.reshape(batch_size, -1)
    s = torch.sum(r2d * G2d, dim=1) / torch.sum(r2d * r2d, dim=1)
    s = s.unsqueeze(dim=1).unsqueeze(dim=2)
    
    ### eval: max 5 pixels error...
    '''
    reg2d = reg2d * s + cent2d
    print(reg2d - gt2d)
    # eval: make sure joint locations matching
    # 2D not very accurate, consider 3D?
    # gt3d and gt2d spatially aligned, roughly s * gt3d[:,:,:2] + t = gt2d
    
    gt2d = torch.cat((gt2d,
        torch.zeros((batch_size, gt2d.shape[1], 1), dtype=gt2d.dtype, device=gt2d.device)
    ), dim=2)
    reg2d = torch.cat((reg2d,
        torch.zeros((batch_size, reg2d.shape[1], 1), dtype=reg2d.dtype, device=reg2d.device)
    ), dim=2)
    
    test_case = range(10)
    for i in test_case:
        np.savetxt('_test_cache/2d_gt_{}.xyz'.format(i), gt2d[i].detach().cpu().numpy(), delimiter=' ')
        np.savetxt('_test_cache/2d_reg_{}.xyz'.format(i), reg2d[i].detach().cpu().numpy(), delimiter=' ')
    '''
    
    # Wrap-up the result into a function, keep z component as a depth inidcator
    def transform(x):
        return torch.bmm(x - cent_J, R) * s + cent3d
        
    return transform
