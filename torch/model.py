import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num


class SNConv2WithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConv2WithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            return self.activation(x)
        return x

class Discriminator2D(nn.Module):
    def __init__(self, nf_in, nf, patch_size, image_dims, patch, use_bias, disc_loss_type='vanilla'):
        nn.Module.__init__(self)
        self.use_bias = use_bias
        approx_receptive_field_sizes = [4, 10, 22, 46, 94, 190, 382, 766]
        num_layers = len(approx_receptive_field_sizes)
        if patch:
            for k in range(len(approx_receptive_field_sizes)):
                if patch_size < approx_receptive_field_sizes[k]:
                    num_layers = k
                    break
        assert(num_layers >= 1)
        self.patch = patch
        self.nf = nf
        dim = min(image_dims[0], image_dims[1])
        num = int(math.floor(math.log(dim, 2)))
        num_layers = min(num, num_layers)
        activation = None if num_layers == 1 else torch.nn.LeakyReLU(0.2, inplace=True)
        self.discriminator_net = torch.nn.Sequential(
            SNConv2WithActivation(nf_in, 2*nf, 4, 2, 1, activation=activation, bias=self.use_bias),
        )
        if num_layers > 1:
            activation = None if num_layers == 2 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p1', SNConv2WithActivation(2*nf, 4*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        if num_layers > 2:
            activation = None if num_layers == 3 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p2', SNConv2WithActivation(4*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        for k in range(3, num_layers):
            activation = None if num_layers == k+1 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p%d' % k, SNConv2WithActivation(8*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        self.final = None
        if not patch or disc_loss_type != 'hinge': #hack
            self.final = torch.nn.Conv2d(nf*8, 1, 1, 1, 0)        
        num_params = count_num_model_params(self.discriminator_net)
        print('#params discriminator', count_num_model_params(self.discriminator_net))
        
        self.compute_valid = None
        if patch:
            self.compute_valid = torch.nn.Sequential(
                torch.nn.AvgPool2d(4, stride=2, padding=1),
            )
            for k in range(1, num_layers):
                self.compute_valid.add_module('p%d' % k, torch.nn.AvgPool2d(4, stride=2, padding=1))
    
    def compute_valids(self, valid):
        if self.compute_valid is None:
            return None
        valid = self.compute_valid(valid)
        return valid

    def forward(self, x, alpha=None):
        for k in range(len(self.discriminator_net)-1):
            x = self.discriminator_net[k](x)
        x = self.discriminator_net[-1](x) 
        
        if self.final is not None:
            x = self.final(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class Conv3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bn=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), bn_before=False, norm_type='batch'):
        super(Conv3, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.bn = None 
        if bn and norm_type != 'none':
            if norm_type == 'batch':
                self.bn = torch.nn.BatchNorm3d(out_channels, momentum=0.8)
            elif norm_type == 'inst':
                self.bn = torch.nn.InstanceNorm3d(out_channels)#, momentum=0.8)
            else:
                raise
        self.bn_before = bn_before

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, verbose=False):    
        x = self.conv3d(x)
        if verbose:
            print(' (conv3)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<1]).item())
            for b in range(x.shape[0]):
                print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        if self.bn_before and self.bn is not None:
            x = self.bn(x)
            if verbose:
                print(' (conv3-bn)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<=1]).item())
                for b in range(x.shape[0]):
                    print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        if self.activation is not None:
            x = self.activation(x)
            if verbose:
                print(' (conv3-act)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<=1]).item())
                for b in range(x.shape[0]):
                    print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        if not self.bn_before and self.bn is not None:
            x = self.bn(x)
            if verbose:
                print(' (conv3-bn)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<=1]).item())
                for b in range(x.shape[0]):
                    print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        return x

class Generator(nn.Module):
    def __init__(self, nf_in_geo, nf_in_color, nf, pass_geo_feats, max_data_size, truncation, max_dilation=1):
        nn.Module.__init__(self)
        self.data_dim = 3
        self.nf = nf
        self.input_mask = nf_in_color > 3
        self.max_data_size = np.array(max_data_size)
        self.use_bias = True
        self.pass_geo_feats = pass_geo_feats
        self.max_dilation = max_dilation
        self.truncation = truncation
        self.interpolate_mode = 'nearest'

        use_dilations = True
        nz_in = max_data_size[0]
        kz = [1] * 34 if nz_in == 1 else [5, 4,3, 4,3,3, 3,3,3,3,3, 3, 3,3, 3,3,3, 5, 4,4, 3,3,3, 3,3,3,3, 3,3, 3,3, 3,3,3]
        dz = [1] * 34 if (not use_dilations or nz_in == 1) else [min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation), min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation)]
        dyx = [1] * 34 if not use_dilations else [min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation), min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation)]
        
        # === geo net === 
        self.geo_0 = nn.Sequential(
                nn.Conv3d(nf_in_geo, self.nf//2, (kz[0], 5, 5), 1, 2, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf//2),
                nn.Conv3d(self.nf//2, self.nf, (kz[1], 4, 4), 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf),
                nn.Conv3d(self.nf, self.nf, (kz[2], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf)
        )
        self.geo_1 = nn.Sequential(
                nn.Conv3d(self.nf, 2*self.nf, (kz[3], 4, 4), 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[4], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[5], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[6], 3, 3), 1, dyx[0], dilation=dyx[0], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[7], 3, 3), 1, dyx[1], dilation=dyx[1], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[8], 3, 3), 1, dyx[2], dilation=dyx[2], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[9], 3, 3), 1, dyx[3], dilation=dyx[3], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[10], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[11], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
        )
        #up
        self.geo_2 = torch.nn.Sequential(
                torch.nn.Conv3d(2*self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf),
                torch.nn.Conv3d(self.nf, self.nf, (kz[12], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf)
        )
        self.geo_occ = torch.nn.Sequential(
                torch.nn.Conv3d(self.nf, self.nf//2, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf//2),
                torch.nn.Conv3d(self.nf//2, 1, (kz[12], 3, 3), 1, 1, bias=self.use_bias)
        )
        self.geo_3 = torch.nn.Sequential(
                torch.nn.Conv3d(self.nf, self.nf//2, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf//2),
                torch.nn.Conv3d(self.nf//2, self.nf//2, (kz[12], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf//2),
                torch.nn.Conv3d(self.nf//2, 1, (kz[12], 3, 3), 1, 1, bias=self.use_bias)
        )
        # === coarse net === 
        self.coarse_0 = nn.Sequential(
                nn.Conv3d(nf_in_color, self.nf, (kz[0], 5, 5), 1, 2, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf),
                nn.Conv3d(self.nf, 2*self.nf, (kz[1], 4, 4), 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                nn.Conv3d(2*self.nf, 2*self.nf, (kz[2], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf)
        )
        self.coarse_0_geo = None
        if self.pass_geo_feats:
            self.coarse_0_geo = nn.Sequential(
                nn.Conv3d(self.nf, self.nf, (kz[1], 4, 4), 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf),
            )
        nf1 = 2*self.nf if not self.pass_geo_feats else 3*self.nf
        nf_factor = 4
        self.coarse_1 = nn.Sequential(
                nn.Conv3d(nf1, nf_factor*self.nf, (kz[3], 4, 4), 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[4], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[5], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[6], 3, 3), 1, dyx[0], dilation=dyx[0], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[7], 3, 3), 1, dyx[1], dilation=dyx[1], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[8], 3, 3), 1, dyx[2], dilation=dyx[2], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[9], 3, 3), 1, dyx[3], dilation=dyx[3], bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[10], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
                
                nn.Conv3d(nf_factor*self.nf, nf_factor*self.nf, (kz[11], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nf_factor*self.nf),
        )
        #up
        self.coarse_2 = torch.nn.Sequential(
                torch.nn.Conv3d(nf_factor*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf),
                torch.nn.Conv3d(2*self.nf, 2*self.nf, (kz[12], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*self.nf)
        )
        self.coarse_3 = torch.nn.Sequential(
                torch.nn.Conv3d(2*self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf),
                torch.nn.Conv3d(self.nf, self.nf//2, (kz[12], 3, 3), 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(self.nf//2),
                torch.nn.Conv3d(self.nf//2, 3, (kz[12], 3, 3), 1, 1, bias=self.use_bias)
        )
        
        #nfr = self.nf//2
        nfr = min(8, self.nf//2) if self.nf > 8 else min(6,self.nf//2)
        print('nf_factor = %d' % nf_factor)
        print('nfr = %d' % nfr)
        d = max(max_dilation, 2)
        nfr0 = nf_in_color
        self.refine_0 = nn.Sequential(
                nn.Conv3d(nfr0, nfr, 5, 1, 2, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nfr),
                nn.Conv3d(nfr, nfr, 3, 1, padding=d, dilation=d, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nfr)
        )
        d = max(max_dilation, 4)
        if self.nf <= 8:
            self.refine_1 = nn.Sequential(
                    nn.Conv3d(nfr, 2*nfr, 3, 1, padding=d, dilation=d, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(2*nfr),
                    
                    nn.Conv3d(2*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, dyx[4], dilation=dyx[4], bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, dyx[5], dilation=dyx[5], bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr)
            )
        else:
            self.refine_1 = nn.Sequential(
                    nn.Conv3d(nfr, 2*nfr, 3, 1, padding=d, dilation=d, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(2*nfr),
                    
                    nn.Conv3d(2*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, dyx[4], dilation=dyx[4], bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, dyx[5], dilation=dyx[5], bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, dyx[6], dilation=dyx[6], bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, dyx[7], dilation=dyx[7], bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
            )
        if self.nf <= 8:
            self.refine_upsample = nn.Sequential(
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr)
            )
        else:
            self.refine_upsample = nn.Sequential(
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
                    nn.Conv3d(nf_factor*nfr, nf_factor*nfr, 3, 1, 1, bias=self.use_bias),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm3d(nf_factor*nfr),
            )
        self.refine_upsample_0 = nn.Sequential(
                nn.Conv3d(nf_factor*nfr, 2*nfr, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*nfr),
                nn.Conv3d(2*nfr, 2*nfr, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(2*nfr),
        )
        self.refine_upsample_1 = nn.Sequential(
                nn.Conv3d(2*nfr, nfr, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nfr),
                nn.Conv3d(nfr, nfr//2, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm3d(nfr//2),
                nn.Conv3d(nfr//2, 3, 1, 1, 0, bias=self.use_bias)
        )
        
        num_params_geo = count_num_model_params(self.geo_0) + count_num_model_params(self.geo_1) + count_num_model_params(self.geo_2) + count_num_model_params(self.geo_occ) + count_num_model_params(self.geo_3)
        num_params_coarse = count_num_model_params(self.coarse_0) + count_num_model_params(self.coarse_1) + count_num_model_params(self.coarse_2) + count_num_model_params(self.coarse_3)
        num_params_refine = count_num_model_params(self.refine_0) + count_num_model_params(self.refine_1) + count_num_model_params(self.refine_upsample) + count_num_model_params(self.refine_upsample_0) + count_num_model_params(self.refine_upsample_1)
        print('#params(geo) = ', num_params_geo)
        print('#params(coarse) = ', num_params_coarse)
        print('#params(refine) = ', num_params_refine)
        print('#params(total) = ', num_params_coarse+num_params_refine+num_params_geo)
        

    def update_sizes(self, input_max_dim):
        self.max_data_size = input_max_dim

    def forward(self, x, mask, pred_color, pred_sdf):        
        if self.input_mask:
            x = torch.cat([x, mask], 1)
            x_geo = x[:,:1,:,:,:]
            mask = x[:,4:,:,:,:]
        else:
            x_geo = x[:,:1,:,:,:]
        x_geo[torch.abs(x_geo) >= self.truncation-0.01] = 0
        
        scale_factor = 2 if self.max_data_size[0] > 1 else (1,2,2)
                
        x_geo = self.geo_0(x_geo)
        x_geo = self.geo_1(x_geo)
        x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_geo = self.geo_2(x_geo)
        x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_occ = self.geo_occ(x_geo)
        x_sdf = self.geo_3(x_geo)

        if pred_color:
            x_color = x[:,1:4,:,:,:]
            x_color = x_color*2-1
            if self.input_mask:
                masked_x = x_color * (1 - mask) + mask
                coarse_x = self.coarse_0(torch.cat((masked_x, mask), dim=1))
            else:
                coarse_x = self.coarse_0(x_color)            
            if self.pass_geo_feats:
                x_geo = self.coarse_0_geo(x_geo)
                coarse_x = torch.cat((coarse_x, x_geo), dim=1)
            coarse_x = self.coarse_1(coarse_x)
            coarse_x = torch.nn.functional.interpolate(coarse_x, scale_factor=scale_factor, mode=self.interpolate_mode)
            coarse_x = self.coarse_2(coarse_x)
            coarse_x = torch.nn.functional.interpolate(coarse_x, scale_factor=scale_factor, mode=self.interpolate_mode)
            coarse_x = self.coarse_3(coarse_x)
            coarse_x = torch.clamp(coarse_x, -1., 1.)
            
            if self.input_mask:
                masked_x = x_color * (1 - mask) + coarse_x * mask
                out = self.refine_0(torch.cat((masked_x, mask), dim=1))
            else:
                out = self.refine_0(coarse_x)
            out = self.refine_1(out)
            out = self.refine_upsample(out)
            out = self.refine_upsample_0(out)
            out = self.refine_upsample_1(out)
            out = torch.clamp(out, -1., 1.)
        else:
            coarse_x = None
            out = None
        
        return x_occ, x_sdf, out








