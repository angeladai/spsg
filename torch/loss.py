
import numpy as np
import torch
import torch.nn.functional as F

import data_util


def compute_targets(target, truncation, use_loss_masking, known, colors):
    assert(len(target.shape) == 5)
    target_for_sdf = data_util.preprocess_sdf_pt(target, truncation)
    target_for_colors = colors
    return target_for_sdf, target_for_colors

def apply_log_transform(sdf):
    sgn = torch.sign(sdf)
    out = torch.log(torch.abs(sdf) + 1)
    out = sgn * out
    return out

def compute_dense_occ(locs, sdfs, dims, truncation):
    assert(len(dims) == 5)
    occ = torch.zeros(dims, dtype=torch.bool).to(sdfs.device)
    occ[locs[:,-1],0,locs[:,0],locs[:,1],locs[:,2]] = torch.abs(sdfs) < truncation-0.01
    return occ

def compute_dense_geo_weights(target_sdf, input_occ, truncation, weight_surf_geo, weight_missing_geo):
    weights = torch.ones_like(target_sdf)
    if weight_surf_geo > 1:
        weights[torch.abs(target_sdf) < truncation-0.01] = weight_surf_geo
    if weight_missing_geo > 1:
        weights[(torch.abs(target_sdf) < truncation-0.01) & (~input_occ)] = weight_missing_geo
    return weights

def compute_color_inpaint_loss(target_colors, output_coarse_color, output_color, known_mask, weight_color):
    target = target_colors.float() / 255.0
    mask = (target[:,:,:,:,0] > 0) | (target[:,:,:,:,1] > 0) | (target[:,:,:,:,2] > 0)
    if known_mask is not None:
        mask = mask & known_mask
    target = target*2 - 1
    
    dims = target.shape[1:4]
    locs_coarse = output_coarse_color[0][:,-1] * dims[0]*dims[1]*dims[2] + output_coarse_color[0][:,0] * dims[1]*dims[2] + output_coarse_color[0][:,1] * dims[2] + output_coarse_color[0][:,2]
    locs = output_color[0][:,-1] * dims[0]*dims[1]*dims[2] + output_color[0][:,0] * dims[1]*dims[2] + output_color[0][:,1] * dims[2] + output_color[0][:,2]
    target_for_coarse = target.view(-1,3)[locs_coarse]
    target_for_refine = target.view(-1,3)[locs]
    output_coarse = output_coarse_color[1]
    output_refine = output_color[1]
    weight_coarse = None
    weight_refine = None
    if weight_color is not None:
        weight_coarse = weight_color.view(-1)[locs_coarse].unsqueeze(1).repeat(1,3)
        weight_refine = weight_color.view(-1)[locs].unsqueeze(1).repeat(1,3)
    coarse_mask = mask.view(-1)[locs_coarse]
    target_for_coarse = target_for_coarse[coarse_mask]
    output_coarse = output_coarse[coarse_mask]
    refine_mask = mask.view(-1)[locs]
    target_for_refine = target_for_refine[refine_mask]
    output_refine = output_refine[refine_mask]
    if weight_color is not None:
        weight_coarse = weight_coarse[coarse_mask]
        weight_refine = weight_refine[refine_mask]
    
    loss_coarse = torch.abs(target_for_coarse - output_coarse)
    loss = torch.abs(target_for_refine - output_refine)
    
    if weight_color is not None:
        loss_coarse = loss_coarse * weight_coarse
        loss = loss * weight_refine
    
    loss_coarse = torch.mean(loss_coarse)
    loss = torch.mean(loss)
    
    return loss_coarse + loss

def compute_geo_loss(target_sdf, output_coarse_sdf, output_sdf, known_mask, weight, log_weight_sdf):
    return compute_geo_loss_dense(target_sdf, output_coarse_sdf, output_sdf, known_mask, weight, log_weight_sdf)

def compute_geo_loss_dense(target_sdf, output_coarse_sdf, output_sdf, known_mask, weight, log_weight_sdf):
    targets = target_sdf if known_mask is None else target_sdf[known_mask]
    output_coarse = None
    if output_coarse_sdf is not None:
        output_coarse = output_coarse_sdf if known_mask is None else output_coarse_sdf[known_mask]
    output_refine = output_sdf if known_mask is None else output_sdf[known_mask]
    
    if log_weight_sdf:
        targets = apply_log_transform(targets)
        output_refine = apply_log_transform(output_refine)
        if output_coarse is not None:
            output_coarse = apply_log_transform(output_coarse)
    
    loss_coarse = 0.0
    if output_coarse_sdf is not None:
        loss_coarse = torch.abs(targets - output_coarse)
    loss = torch.abs(targets - output_refine)
    
    if weight is not None:
        w = weight if known_mask is None else weight[known_mask]
        if output_coarse_sdf is not None:
            loss_coarse = loss_coarse * w
        loss = loss * w
    
    if output_coarse_sdf is not None:
        loss_coarse = torch.mean(loss_coarse)
    loss = torch.mean(loss)
    
    return loss_coarse + loss

def compute_geo_occ_loss(target_sdf, output_occ, known_mask, weight, truncation):
    if isinstance(output_occ, tuple):
        loss = compute_geo_occ_loss_dense(target_sdf, output_occ[0], known_mask, weight, truncation)
        factor = 4
        for k in range(1, len(output_occ)):
            factor = factor // 2
            if output_occ[k] is not None:
                loss += compute_geo_occ_loss_sparse(target_sdf, output_occ[k], known_mask, weight, truncation, factor)
    else:
        loss = compute_geo_occ_loss_dense(target_sdf, output_occ, known_mask, weight, truncation)
    return loss

def compute_geo_occ_loss_dense(target_sdf, output_occ, known_mask, weight, truncation):
    target = (torch.abs(target_sdf) < truncation).float()
    known = known_mask
    w = weight
    if output_occ.shape[2] != target.shape[2]:
        factor = target.shape[2] // output_occ.shape[2]
        target = torch.nn.MaxPool3d(factor)(target)
        if known is not None:
            known = torch.nn.MaxPool3d(factor)(known.float()).bool()
        if w is not None:
            w = torch.nn.MaxPool3d(factor)(w)
    target = target if known is None else target[known]
    pred = output_occ if known is None else output_occ[known]
    if w is not None and known is not None:
        w = w[known]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, weight=w)
    return loss

def compute_geo_occ_loss_sparse(target_sdf, output_occ, known_mask, weight, truncation, factor):
    target = (torch.abs(target_sdf) < truncation).float()
    known = known_mask
    w = weight
    pred = output_occ[1].view(-1)
    if factor > 1:
        target = torch.nn.MaxPool3d(factor)(target)
        if known is not None:
            known = torch.nn.MaxPool3d(factor)(known.float()).bool()
        if w is not None:
            w = torch.nn.MaxPool3d(factor)(w)    
    dims = target.shape[2:]
    locs = output_occ[0][:,-1] * dims[0]*dims[1]*dims[2] + output_occ[0][:,0] * dims[1]*dims[2] + output_occ[0][:,1] * dims[2] + output_occ[0][:,2]
    target = target.view(-1)[locs]
    if w is not None:
        w = w.view(-1)[locs]
    if known is not None:
        known = known.view(-1)[locs]
        target = target[known]
        pred = pred[known]
        w = w[known]    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, weight=w)
    return loss

def compute_iou_occ(target_sdf, output_occ, known_mask, truncation):
    if isinstance(output_occ, tuple):
        iou = 0.0
        norm = 0.0
        factor = 4
        cur_iou = compute_iou_occ_dense(target_sdf, output_occ[0], known_mask, truncation)
        if cur_iou >= 0:
            iou += cur_iou
            norm += 1
        for k in range(1, len(output_occ)):
            factor = factor // 2
            if output_occ[k] is not None:
                cur_iou = compute_iou_occ_sparse(target_sdf, output_occ[k], known_mask, truncation, factor)
                if cur_iou >= 0:
                    iou += cur_iou
                    norm += 1
        iou /= norm
    else:
        iou = compute_iou_occ_dense(target_sdf, output_occ, known_mask, truncation)
    return iou

def compute_iou_occ_sparse(target_sdf, output_occ, known_mask, truncation, factor):
    target = (torch.abs(target_sdf) < truncation)
    known = known_mask
    pred_locs = output_occ[0][torch.nn.Sigmoid()(output_occ[1].detach().view(-1)) > 0.5]
    if factor > 1:
        target = torch.nn.MaxPool3d(factor)(target.float()).bool()
        if known is not None:
            known = torch.nn.MaxPool3d(factor)(known.float()).bool()
    pred = torch.zeros(target.shape, dtype=torch.bool).to(output_occ[1].device)
    pred[pred_locs[:,-1], 0, pred_locs[:,0], pred_locs[:,1], pred_locs[:,2]] = 1
    if known is not None:
        target = target[known]
        pred = pred[known]
    corr = torch.sum(pred & target).item()
    union = torch.sum(pred | target).item()
    if union == 0:
        return -1
    return float(corr)/float(union)

def compute_iou_occ_dense(target_sdf, output_occ, known_mask, truncation):
    target = torch.abs(target_sdf) < truncation
    known = known_mask
    if output_occ.shape[2] != target.shape[2]:
        factor = target.shape[2] // output_occ.shape[2]
        target = torch.nn.MaxPool3d(factor)(target.float()).bool()
        if known is not None:
            known = torch.nn.MaxPool3d(factor)(known.float()).bool()
    target = target if known is None else target[known]
    pred = output_occ if known is None else output_occ[known]
    pred = torch.nn.Sigmoid()(pred) > 0.5
    corr = torch.sum(pred & target).item()
    union = torch.sum(pred | target).item()
    if union == 0:
        return -1
    return float(corr)/float(union)

def compute_iou_sdf(target_sdf, output_sdf, output_occ):
    tgt = torch.abs(target_sdf) < 1
    pred = torch.abs(output_sdf) < 1
    if output_occ is not None:
        pred = pred & output_occ
    intersection = torch.sum(pred & tgt).item()
    union = torch.sum(pred | tgt).item()
    if union == 0:
        return -1
    return float(intersection)/float(union)

def compute_2dcolor_loss(raycast_color, target_color, weight_color):
    valid = raycast_color != -float('inf')
    pred = raycast_color
    tgt = target_color
    if weight_color is not None:
        w = weight_color.view(weight_color.shape[0], weight_color.shape[2], weight_color.shape[3], 1)
        pred = pred * w
        tgt = tgt * w
    pred = pred[valid]
    tgt = tgt[valid]
    loss_color = torch.mean(torch.abs(pred - tgt))
    return loss_color

# z-y-x (note: returns unnormalized!)
def compute_normals_dense(sdf):
    assert(len(sdf.shape) == 5) # batch mode
    dims = sdf.shape[2:]
    sdfx = sdf[:,:,1:dims[0]-1,1:dims[1]-1,2:dims[2]] - sdf[:,:,1:dims[0]-1,1:dims[1]-1,0:dims[2]-2]
    sdfy = sdf[:,:,1:dims[0]-1,2:dims[1],1:dims[2]-1] - sdf[:,:,1:dims[0]-1,0:dims[1]-2,1:dims[2]-1]
    sdfz = sdf[:,:,2:dims[0],1:dims[1]-1,1:dims[2]-1] - sdf[:,:,0:dims[0]-2,1:dims[1]-1,1:dims[2]-1]
    return torch.cat([sdfx, sdfy, sdfz], 1)

def compute_normals(sdf, sdf_locs, transform=None):
    normals = compute_normals_dense(sdf)
    normals = torch.nn.functional.pad(normals, (1,1,1,1,1,1),value=-float('inf'))
    normals = normals[sdf_locs[:,3],:,sdf_locs[:,0],sdf_locs[:,1],sdf_locs[:,2]].contiguous()
    mask = normals[:,0] != -float('inf')
    normals[normals == -float('inf')] = 0
    if transform is not None:
        n = []
        for b in range(transform.shape[0]):
            n.append(torch.matmul(transform[b,:3,:3], normals[sdf_locs[:,-1] == b].t()).t())
        normals = torch.cat(n)
    normals = -torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-5, out=None)
    return normals

def compute_normals_sparse(sdf_locs, sdf_vals, dims, transform=None):
    dims = [dims[0], dims[1], dims[2]]
    batch_size = sdf_locs[-1,-1]+1
    sdf = torch.zeros(batch_size, 1, dims[0], dims[1], dims[2]).to(sdf_vals.device)
    sdf[sdf_locs[:,-1], :, sdf_locs[:,0], sdf_locs[:,1], sdf_locs[:,2]] = sdf_vals
    #sdf = scn.SparseToDense(3, sdf_vals.shape[1])(scn.InputLayer(3, dims, mode=0)([sdf_locs, sdf_vals]))
    normals = compute_normals_dense(sdf)
    normals = torch.nn.functional.pad(normals, (1,1,1,1,1,1),value=-float('inf'))
    normals = normals[sdf_locs[:,3],:,sdf_locs[:,0],sdf_locs[:,1],sdf_locs[:,2]].contiguous()
    mask = normals[:,0] != -float('inf')
    normals[normals == -float('inf')] = 0
    if transform is not None:
        n = []
        for b in range(transform.shape[0]):
            #n.append(normals[sdf_locs[:,-1] == b])
            n.append(torch.matmul(transform[b,:3,:3], normals[sdf_locs[:,-1] == b].t()).t())
            #bmask = (sdf_locs[:,-1] == b) & mask
            #normals[bmask] = torch.matmul(transform[b,:3,:3], normals[bmask].t()).t()
        normals = torch.cat(n)
    #normals[mask] = -torch.nn.functional.normalize(normals[mask], p=2, dim=1, eps=1e-5, out=None)
    normals = -torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-5, out=None)
    return normals

class ImageGradient(torch.nn.Module):
    def __init__(self):
        super(ImageGradient, self).__init__()
        kernel = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)/8.0
        self.conv_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = torch.nn.Parameter(kernel)
        
        kernel = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)/8.0
        self.conv_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = torch.nn.Parameter(kernel)
        
    def forward(self, image):
        dx_r = self.conv_x(image[:,:1,:,:])
        dx_g = self.conv_x(image[:,1:2,:,:])
        dx_b = self.conv_x(image[:,2:,:,:])
        dy_r = self.conv_y(image[:,:1,:,:])
        dy_g = self.conv_y(image[:,1:2,:,:])
        dy_b = self.conv_y(image[:,2:,:,:])
        #g = torch.sqrt(torch.pow(dx,2)+ torch.pow(dy,2))        
        return torch.cat([dx_r, dx_g, dx_b, dy_r, dy_g, dy_b], 1)

def filter_proj_target(raycast_color, color_thresh, color_space):
    if color_space == 'rgb':
        thresh = 0.01 if color_space == 'lab' else float(color_thresh)/255.0
        invalid = (raycast_color[:,:,:,0] < thresh) & (raycast_color[:,:,:,1] < thresh) & (raycast_color[:,:,:,2] < thresh)
    elif color_space == 'lab':
        thresh = 0.01
        invalid = (raycast_color[:,:,:,0] < thresh) & (raycast_color[:,:,:,1] > 0.5-thresh) & (raycast_color[:,:,:,1] < 0.5+thresh) & (raycast_color[:,:,:,2] > 0.5-thresh) & (raycast_color[:,:,:,2] < 0.5+thresh)
    else:
        raise
    return invalid

def compute_missing_geo_mask(input_occ, target_sdf, truncation):    
    occ = torch.nn.MaxPool3d(kernel_size=8)(input_occ.float())
    occ = torch.nn.Upsample(scale_factor=8)(occ)
    
    occ = occ.bool()
    mask = torch.abs(target_sdf) < truncation
    mask[occ] = 0
    
    return mask

class GANLoss:

    def __init__(self, loss_type):
        self.compute_generator_loss = GANLoss.compute_generator_loss_wasserstein
        self.compute_discriminator_loss = None
        if loss_type == 'hinge':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_hinge
        elif loss_type == 'vanilla':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_vanilla
            self.compute_generator_loss = GANLoss.compute_generator_loss_vanilla
        elif loss_type == 'wgan':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_wasserstein
        elif loss_type == 'wgan_gp':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_wasserstein_gp

    @staticmethod
    def compute_discriminator_loss_vanilla(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        d_real = ref_disc(in_real, alpha)
        d_fake = ref_disc(in_fake, alpha)
        if weight is not None:
            d_real = d_real * weight.view(d_real.shape)
            d_fake = d_fake * weight.view(d_fake.shape)        
        if valid is not None:
            d_real = d_real[valid]
            d_fake = d_fake[valid]
        real_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_real, torch.ones(d_real.shape).cuda()*label_smoothing_factor, reduction='none')
        real_loss = torch.mean(real_loss, 1)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_fake, torch.zeros(d_fake.shape).cuda(), reduction='none')
        fake_loss = torch.mean(fake_loss, 1)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)
        
    @staticmethod
    def compute_generator_loss_vanilla(ref_disc, in_fake, alpha=1):
        d_fake = ref_disc(in_fake, alpha)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_fake, torch.ones(d_fake.shape).cuda())
        return fake_loss
        
    @staticmethod
    def compute_generator_loss_wasserstein(ref_disc, in_fake, alpha=1):
        return -torch.mean(ref_disc(in_fake, alpha))

    @staticmethod
    def compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        d_real = ref_disc(in_real, alpha)
        d_fake = ref_disc(in_fake, alpha)
        if weight is not None:
            d_real = d_real * weight.view(d_real.shape)
            d_fake = d_fake * weight.view(d_fake.shape)  
        if valid is not None:
            d_real = d_real[valid]
            d_fake = d_fake[valid]
        real_loss = -torch.mean(d_real,1)
        fake_loss = torch.mean(d_fake,1)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)

    @staticmethod
    def compute_discriminator_loss_hinge(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        d_real = ref_disc(in_real, alpha)
        d_fake = ref_disc(in_fake, alpha)
        if weight is not None:
            d_real = d_real * weight.view(d_real.shape)
            d_fake = d_fake * weight.view(d_fake.shape)         
        if valid is not None:
            d_real = d_real[valid]
            d_fake = d_fake[valid]
        real_loss = torch.mean(torch.nn.functional.relu(1. - d_real), 1)
        fake_loss = torch.mean(torch.nn.functional.relu(1. + d_fake), 1)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)

    @staticmethod
    def compute_discriminator_loss_wasserstein_gp(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        if not val_mode:
            real_loss, fake_loss, _ = GANLoss.compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake, valid, weight, val_mode, alpha)
            return real_loss, fake_loss, GANLoss.compute_gradient_penalty(ref_disc, in_real, in_fake)
        else:
            return GANLoss.compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake, valid, weight, val_mode, alpha)

    @staticmethod
    def compute_gradient_penalty(ref_disc, in_real, in_fake):
        # Calculate interpolation
        if len(in_real.shape) == 5:
            alpha = torch.rand(in_real.shape[0], 1, 1, 1, 1)
        else:
            alpha = torch.rand(in_real.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(in_real)
        alpha = alpha.cuda(in_real.device.index)
        interpolated = (alpha * in_real + (1 - alpha) * in_fake).requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = ref_disc(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).cuda(in_real.device.index), create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(in_real.shape[0], -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
