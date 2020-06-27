
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_util

def compute_view_matrix(world2grids, poses):
    batch_size = world2grids.shape[0]
    num_frames = poses.shape[0]//batch_size
    if num_frames == 1:
        view_matrix = torch.matmul(world2grids.cuda(), poses)
    else:
        view_matrix = torch.matmul(world2grids.cuda().unsqueeze(1).repeat(1, num_frames, 1, 1).view(-1, 4, 4), poses)
    return view_matrix

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(b * c * d) #G.div(a * b * c * d)

def preprocess_rendered_target_images(render_images, target_images, mask=None):
    invalid_mask = render_images == -float('inf')
    render_images[invalid_mask] = target_images[invalid_mask]
    if mask is not None:
        render_images[mask == 0] = target_images[mask == 0]    
    return render_images, target_images

def compute_imagel1_loss(output_color, target_color, mask=None):
    output_color, target_color = preprocess_rendered_target_images(output_color, target_color, mask)
    loss = torch.mean(torch.abs(output_color - target_color))
    return loss

def compute_style_loss(output_color, target_color, model_style, compute_style, compute_content, mask=None):
    # mask out any non-existent geometry in the frame
    output_color, target_color = preprocess_rendered_target_images(output_color, target_color, mask)
    target_features = model_style(target_color)
    output_features = model_style(output_color)
    loss = 0.0
    loss_content = 0.0
    for k in range(len(output_features)):
        if compute_content:
            loss_content += F.mse_loss(output_features[k], target_features[k])
        if compute_style:
            tgt = gram_matrix(target_features[k])
            pred = gram_matrix(output_features[k])
            loss += F.mse_loss(pred*10, tgt*10)
    return loss, loss_content

# for debugging
def compute_style_loss_layers(output_color, target_color, model_style, compute_style, compute_content, mask=None):
    # mask out any non-existent geometry in the frame
    output_color, target_color = preprocess_rendered_target_images(output_color, target_color, mask)

    target_features = model_style(target_color)
    output_features = model_style(output_color)
    loss_style = []
    loss_content = []
    for k in range(len(output_features)):
        if compute_content:
            loss_content.append(F.mse_loss(output_features[k], target_features[k]).item())
        if compute_style:
            tgt = gram_matrix(target_features[k])
            pred = gram_matrix(output_features[k])
            loss_style.append(F.mse_loss(pred*10, tgt*10).item())
    return loss_style, loss_content

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num


# for VGG
#style_layers_default = [1, 2, 3, 4, 5]
#style_layers_default = [1, 2, 3, 4]
#style_layers_default = [1, 2, 3]
#style_layers_default = [1, 2]
#style_layers_default = [1]
#style_layers_default = [4]
style_layers_default = [8]
#style_layers_default = [12]
#style_layers_default = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class Model(nn.Module):
    def __init__(self, vgg, normalization_mean, normalization_std,
                    style_layers=style_layers_default,
                    to_cuda=True):
        super(Model, self).__init__()
        self.style_layers = np.sort(style_layers)
        max_conv_layer = self.style_layers[-1]
    
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std)
        if to_cuda:
            normalization.cuda()

        self.modules = [normalization]
    
        i = 0  # increment every time we see a conv
        self.indices = set()
        for layer in vgg.features.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
                self.indices.add(len(self.modules))
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    
            self.modules.append(layer)
    
            if i == max_conv_layer:
                break
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = []
        for i in range(len(self.modules)):
            x = self.modules[i](x)
            if i in self.indices:
                outputs.append(x)
        return outputs


