import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import os
import numpy as np
from cv2 import imread, imwrite, connectedComponents

#####################
#   Initializers    #
#####################

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def init_net(net, init_type='normal', device=torch.device('cuda')):
    init_weights(net, init_type)
    return net.to(device)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        #norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        norm_layer = nn.BatchNorm2d
    elif layer_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer
    
def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = nn.ReLU(inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = nn.LeakyReLU(0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = nn.ELU(inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def define_encoder(im_size, nz, nef, netE, ndown, norm='batch', nl='lrelu', init_type='xavier', device=None):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if netE == 'resnet':
        net = ResNetEncoder(im_size, nz, nef, ndown, norm_layer, nl_layer)
    elif netE == 'vggnet':
        net = VGGEncoder(im_size, nz, nef, ndown, norm_layer, nl_layer)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % netE)

    return init_net(net, init_type, device)
    
def define_decoder(im_size, nz, ndf, netD, nup, norm='batch', nl='lrelu', init_type='xavier', device=None):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)
    if netD == 'convres':
        net = ConvResDecoder(im_size, nz, ndf, nup=nup, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'conv-up':
        net = ConvUpSampleDecoder(im_size, nz, ndf, nup=nup, norm_layer=norm_layer, nl_layer=nl_layer)
    else:
        raise NotImplementedError('Decoder model name [%s] is not recognized' % netD)

    return init_net(net, init_type, device)
    

#####################
#      Losses       #   
#####################

def acquire_weights(UV_weight_npy):
    if os.path.isfile(UV_weight_npy):
        return np.load(UV_weight_npy)
    else:
        mask_name = UV_weight_npy.replace('weights.npy', 'mask.png')
        print(mask_name)
        UV_mask = imread(mask_name)
        if UV_mask.ndim == 3:
            UV_mask = UV_mask[:,:,0]
        ret, labels = connectedComponents(UV_mask, connectivity=4)
        unique, counts = np.unique(labels, return_counts=True)
        print(unique, counts)
        
        UV_weights = np.zeros_like(UV_mask).astype(np.float32)
        for id, count in zip(unique, counts):
            if id == 0:
                continue
            indices = np.argwhere(labels == id)
            UV_weights[indices[:,0], indices[:,1]] = 1 / count
        
        UV_weights *= np.prod(UV_mask.shape)   # adjust loss to [0,10] level.
        np.save(UV_weight_npy, UV_weights)
        return UV_weights
        
        
class WeightedL1Loss(nn.Module):
    def __init__(self, uv_map, device):
        super(WeightedL1Loss, self).__init__()
        self.weight = torch.from_numpy(acquire_weights(
            ('data_utils/{}_UV_weights.npy'.format(uv_map))
        )).to(device)
        print(self.weight.shape)
        self.loss = nn.L1Loss()
    
    def __call__(self, input, target):
        return self.loss(input * self.weight, target * self.weight)
    
class TotalVariationLoss(nn.Module):
    def __init__(self, uv_map, device):
        super(TotalVariationLoss, self).__init__()
        weight = torch.from_numpy(acquire_weights(
            ('data_utils/{}_UV_weights.npy'.format(uv_map))
        )).to(device)
        self.weight = weight[0:-1, 0:-1]
        self.factor = self.weight.shape[0] * self.weight.shape[1]
        
    def __call__(self, input):
        tv = torch.abs(input[:,:,0:-1, 0:-1] - input[:,:,0:-1, 1:]) \
            + torch.abs(input[:,:,0:-1, 0:-1] - input[:,:,1:, 0:-1])
        return torch.sum(tv * self.weight) / self.factor
    
#####################
#      Networks     #
#####################

#####  ResNet  #####

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Use kernel size 4 to make sure deconv(conv(x)) has the same shape as x
# TODO: replace convtrans to upsample to reduce checkerboard artifacts
# not working well...
# https://distill.pub/2016/deconv-checkerboard/
def deconv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
                  kernel_size=3, stride=1, padding=0)
    )
    # return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
    #                           padding=1, bias=False)

# Basic resnet block:
# x ---------------- shortcut ---------------x
# \___conv___norm____relu____conv____norm____/
class BasicResBlock(nn.Module):
    def __init__(self, inplanes, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.LeakyReLU(0.2, True)):
        super(BasicResBlock, self).__init__()

        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.inplanes = inplanes

        layers = [
            conv3x3(inplanes, inplanes),
            norm_layer(inplanes),
            activation_layer,
            conv3x3(inplanes, inplanes),
            norm_layer(inplanes)
        ]
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return self.res(x) + x

# ResBlock: A classic ResBlock with 2 conv layers and a up/downsample conv layer. (2+1)
# x ---- BasicConvBlock ---- ReLU ---- conv/upconv ----
# If direction is "down", we use nn.Conv2d with stride > 1, getting a smaller image
# If direction is "up", we use nn.ConvTranspose2d with stride > 1, getting a larger image
class ConvResBlock(nn.Module):
    def __init__(self, inplanes, planes, direction, stride=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True)):
        super(ConvResBlock, self).__init__()
        self.res = BasicResBlock(inplanes, norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        self.activation = activation_layer

        if stride == 1 and inplanes == planes:
            conv = lambda x: x
        else:
            if direction == 'down':
                conv = conv3x3(inplanes, planes, stride=stride)
            elif direction == 'up':
                conv = deconv3x3(inplanes, planes, stride=stride)
            else:
                raise (ValueError('Direction must be either "down" or "up", get %s instead.' % direction))
        self.conv = conv
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        return self.conv(self.activation(self.res(x)))
        
        #im_size, nz, nef, norm_layer, nl_layer)
class ResNetEncoder(nn.Module):
    def __init__(self, im_size, nz=256, ngf=64, ndown=6,
        norm_layer=None, nl_layer=None):
        super(ResNetEncoder, self).__init__()
        self.ngf = ngf
        fc_dim = 2 * nz

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0),
            norm_layer(ngf),
            nl_layer,
        ]
        prev = 1
        for i in range(ndown):
            im_size //= 2
            cur = min(8, prev*2)
            layers.append(ConvResBlock(ngf * prev, ngf * cur, direction='down', stride=2,
                norm_layer=norm_layer, activation_layer=nl_layer))
            prev = cur

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(im_size * im_size * ngf * cur, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nl_layer,
            nn.Linear(fc_dim, nz)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
        
#####  VGGNet  #####

'''
    This is a replica of torchvision.models.vgg13_bn with modified input size
'''
class VGGEncoder(nn.Module):
    def __init__(self, im_size, nz=256, ngf=64, ndown=5,
        norm_layer=None, nl_layer=None):
        super(VGGEncoder, self).__init__()
        cfg_parts = [
            [1 * ngf, 1 * ngf, 'M'], 
            [2 * ngf, 2 * ngf, 'M'],
            [4 * ngf, 4 * ngf, 'M'],  # [4 * ngf, 4 * ngf, 4 * ngf, 'M'],
            [8 * ngf, 8 * ngf, 'M'],  # [8 * ngf, 8 * ngf, 8 * ngf, 'M'],
        ]
        custom_cfg = []
        for i in range(ndown):
            custom_cfg += cfg_parts[min(i, 3)]
        fc_dim = 4 * nz
        
        self.features = self._make_layers(
            cfg=custom_cfg,
            batch_norm=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
        )
        im_size = im_size // (2**ndown)
        self.avgpool=nn.AdaptiveAvgPool2d((im_size, im_size))
        self.classifier = nn.Sequential(
            nn.Linear(512 * im_size * im_size, fc_dim),
            nl_layer,
            nn.Dropout(),
            nn.Linear(fc_dim, nz),
        )
    
    def _make_layers(self, cfg, batch_norm=False, norm_layer=None, nl_layer=None):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), nl_layer]
                else:
                    layers += [conv2d, nl_layer]
                in_channels = v
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


#####  Decoder #####
class ConvResDecoder(nn.Module):
    '''
        ConvResDecoder: Use convres block for upsampling
    '''
    def __init__(self, im_size, nz, ngf=64, nup=6,
        norm_layer=None, nl_layer=None):
        super(ConvResDecoder, self).__init__()
        self.im_size = im_size // (2 ** nup)
        fc_dim = 2 * nz
        
        layers = []
        prev = 8
        for i in range(nup-1, -1, -1):
            cur = min(prev, 2**i)
            layers.append(ConvResBlock(ngf * prev, ngf * cur, direction='up', stride=2,
                norm_layer=norm_layer, activation_layer=nl_layer))
            prev = cur
        
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(nz, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fc_dim, self.im_size * self.im_size * ngf * 8),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.im_size, self.im_size)
        return self.conv(x)
        
class ConvUpSampleDecoder(nn.Module):
    '''
        SimpleDecoder
    '''
    def __init__(self, im_size, nz, ngf=64, nup=6,
        norm_layer=None, nl_layer=None):
        super(ConvUpSampleDecoder, self).__init__()
        self.im_size = im_size // (2 ** nup)
        fc_dim = 4 * nz
        
        layers = []
        prev = 8
        for i in range(nup-1, -1, -1):
            cur = min(prev, 2**i)
            layers.append(deconv3x3(ngf * prev, ngf * cur, stride=2))
            prev = cur
        
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(nz, fc_dim),
            nl_layer,
            nn.Dropout(),
            nn.Linear(fc_dim, self.im_size * self.im_size * ngf * 8),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.im_size, self.im_size)
        return self.conv(x)
    
if __name__ == '__main__':
    acquire_weights('../data_utils/smpl_fbx_template_UV_weights.npy')
    