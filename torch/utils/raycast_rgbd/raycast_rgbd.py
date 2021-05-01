import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Function

import raycast_rgbd_cuda


class RayCastRGBDFunction(Function):
    @staticmethod
    def forward(ctx, locs, vals_sdf, vals_colors, vals_normals, view_matrix_inv, intrinsic_params, dims3d, width, height, depth_min, depth_max, thresh_sample_dist, ray_increment, image_color, image_depth, image_normal, sparse_mapping, mapping3dto2d, mapping3dto2d_num, d_color, d_depth, d_normal):
        if locs.shape[0] > mapping3dto2d.shape[0]:
            print('ERROR: locs size %s vs mapping3dto2d size %s' % (str(locs.shape), str(mapping3dto2d.shape)))
            locs = locs[:mapping3dto2d.shape[0]]
            vals_sdf = vals_sdf[:mapping3dto2d.shape[0]]
            vals_colors = vals_colors[:mapping3dto2d.shape[0]]
            vals_normals = vals_normals[:mapping3dto2d.shape[0]]
        batch_size = locs[-1,-1] + 1                
        raycast_rgbd_cuda.construct_dense_sparse_mapping(locs, sparse_mapping)
        opts = torch.FloatTensor([width, height, depth_min, depth_max, thresh_sample_dist, ray_increment, dims3d[2], dims3d[1], dims3d[0]])
        raycast_rgbd_cuda.forward(sparse_mapping.to(vals_sdf.device), locs.to(vals_sdf.device), vals_sdf, vals_colors, vals_normals, view_matrix_inv, image_color, image_depth, image_normal, mapping3dto2d, mapping3dto2d_num, intrinsic_params, opts)
        variables = [sparse_mapping, mapping3dto2d, mapping3dto2d_num, torch.IntTensor([batch_size,  dims3d[2], dims3d[1], dims3d[0], locs.shape[0]]), d_color, d_depth, d_normal]
        ctx.save_for_backward(*variables)
        
        return image_color, image_depth, image_normal

    @staticmethod
    def backward(ctx, grad_color, grad_depth, grad_normal):
        sparse_mapping, mapping3dto2d, mapping3dto2d_num, dims, d_color, d_depth, d_normal = ctx.saved_variables;
        raycast_rgbd_cuda.backward(
            grad_color.contiguous(), grad_depth.contiguous(), grad_normal.contiguous(), sparse_mapping, mapping3dto2d, mapping3dto2d_num, dims, d_color, d_depth, d_normal)
        
        return None, d_depth[:dims[4]], d_color[:dims[4]], d_normal[:dims[4]], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class RaycastRGBD(nn.Module):
    def __init__(self, batch_size, dims3d, width, height, depth_min, depth_max, thresh_sample_dist, ray_increment, max_num_frames=1, max_num_locs_per_sample=200000, max_pixels_per_voxel=64):#32):
        super(RaycastRGBD, self).__init__()
        #TODO CAN MAKE THIS TRIVIALLY MORE MEMORY EFFICIENT
        self.dims3d = dims3d
        self.width = width
        self.height = height
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.thresh_sample_dist = thresh_sample_dist
        self.ray_increment = ray_increment
        self.max_num_locs_per_sample = max_num_locs_per_sample
        # pre-allocate raycast_color outputs
        self.image_depth = torch.zeros(batch_size*max_num_frames, height, width).cuda()
        self.image_normal = torch.zeros(batch_size*max_num_frames, height, width, 3).cuda()
        self.image_color = torch.zeros(batch_size*max_num_frames, height, width, 3).cuda()
        self.mapping3dto2d = torch.zeros(batch_size*max_num_frames*max_num_locs_per_sample, max_pixels_per_voxel, dtype=torch.int).cuda() # no color trilerp -> only pixel index here
        self.mapping3dto2d_num = torch.zeros(batch_size*max_num_frames*max_num_locs_per_sample, dtype=torch.int).cuda() # counter for self.mapping3dto2d
        self.sparse_mapping = torch.zeros(batch_size, dims3d[0], dims3d[1], dims3d[2], dtype=torch.int).cuda()
        self.d_color = torch.zeros(batch_size*max_num_locs_per_sample, 3).cuda()
        self.d_normal = torch.zeros(batch_size*max_num_locs_per_sample, 3).cuda()
        self.d_depth = torch.zeros(batch_size*max_num_locs_per_sample, 1).cuda()

    def get_max_num_locs_per_sample(self):
        return self.max_num_locs_per_sample
    
    def forward(self, locs, vals_sdf, vals_colors, vals_normals, view_matrix, intrinsic_params):
        return RayCastRGBDFunction.apply(locs, vals_sdf, vals_colors, vals_normals, view_matrix, intrinsic_params, self.dims3d, self.width, self.height, self.depth_min, self.depth_max, self.thresh_sample_dist, self.ray_increment, self.image_color, self.image_depth, self.image_normal, self.sparse_mapping, self.mapping3dto2d, self.mapping3dto2d_num, self.d_color, self.d_depth, self.d_normal)


class RaycastOcc(nn.Module):
    def __init__(self, batch_size, dims3d, width, height, depth_min, depth_max, ray_increment):
        super(RaycastOcc, self).__init__()
        self.dims3d = dims3d
        self.width = width
        self.height = height
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.ray_increment = ray_increment
        self.occ2d = torch.zeros(batch_size, 1, height, width, dtype=torch.uint8).cuda()

    def forward(self, occ3d, view_matrix, intrinsic_params):
        opts = torch.FloatTensor([self.width, self.height, self.depth_min, self.depth_max, self.ray_increment, self.dims3d[2], self.dims3d[1], self.dims3d[0]])      
        raycast_color_cuda.raycast_occ(occ3d, self.occ2d, view_matrix, intrinsic_params, opts)
        return self.occ2d


