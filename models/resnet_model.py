import torch

from .base_model import BaseModel
from . import networks

class ResNetModel(BaseModel):
    def name(self):
        return 'ResNetModel'
        
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
        
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['L1', 'TV']
        self.model_names = ['encoder', 'decoder']
        
        self.encoder = networks.define_encoder(opt.im_size, opt.nz, opt.nchannels, netE=opt.model, ndown=opt.ndown,
            norm = opt.norm, nl=opt.nl, init_type=opt.init_type, device=self.device)
        
        self.decoder = networks.define_decoder(opt.im_size, opt.nz, opt.nchannels, netD=opt.netD, nup=opt.ndown,
            norm = opt.norm, nl=opt.nl, init_type=opt.init_type, device=self.device)
        
        if opt.phase == 'train':
            self.L1_loss = networks.WeightedL1Loss(opt.uv_prefix, self.device)  # requires a weight npy
            self.TV_loss = networks.TotalVariationLoss(opt.uv_prefix, self.device)
            self.encoder.train()
            self.decoder.train()
            
            self.optimizers = []
            self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_enc)
            
            self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_dec)
        else:
            self.encoder.eval()
            self.decoder.eval()
    
    def set_input(self, input):
        self.real_input = input['im_data']
        self.real_UV = input['uv_data']
    
    '''
        forward: Train one step, return loss calues
    '''
    def train_one_batch(self, data):
        self.set_input(data)
        self.fake_UV = self.decoder(self.encoder(self.real_input))
        l1_loss = self.L1_loss(self.fake_UV, self.real_UV)
        tv_loss = self.TV_loss(self.fake_UV)
        total_loss = l1_loss #+ self.opt.tv_weight * tv_loss
        
        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        total_loss.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        
        return {
            'l1': l1_loss.item(),
            'tv': tv_loss.item(),
            'total': total_loss.item()
        }
    
    def get_current_visuals(self):
        # return: real image, real UV maps fake UV maps
        return {
            'real_image': self.real_input[0],
            'real_UV': self.real_UV[0],
            'fake_UV': self.fake_UV[0]
        }
        
