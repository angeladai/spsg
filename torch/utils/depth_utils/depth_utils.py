import os
import numpy as np
import torch
import PIL.Image
import time
import plyfile

import depth_utils_cuda

def read_mat4_from_file(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return np.asarray(lines).astype(np.float32)
def get_intrinsic_params_from_matrix(intrinsic):
    params = np.array([intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]], dtype=np.float32)
    return params

def visualize_campos(campos, output_file, normals=None, transform=None):
    assert(len(campos.shape) == 3)
    verts = []
    norms = []
    for r in range(campos.shape[0]):
        for c in range(campos.shape[1]):
            if campos[r,c,0] == -float('inf'):
                continue
            if normals is not None and (normals[r,c,0] == -float('inf') or np.all(normals[r,c] == 0)):
                continue
            verts.append(campos[r,c])
            if normals is not None:
                norms.append(normals[r,c])
    verts = np.stack(verts)
    if norms:
        norms = np.stack(norms)
    if transform is not None:
        x = np.ones((verts.shape[0], 4))
        x[:, :3] = verts
        x = np.matmul(transform, np.transpose(x))
        x = np.transpose(x)
        verts = np.divide(x[:, :3], x[:, 3, None])
    if normals is None:
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = plyfile.PlyElement.describe(verts,'vertex')
        plyfile.PlyData([el]).write(output_file)
    else:
        verts = np.concatenate([verts, norms], 1)
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('nx', 'f4'), ('ny', 'f4'),('nz', 'f4')])
        el = plyfile.PlyElement.describe(verts,'vertex')
        plyfile.PlyData([el]).write(output_file)

def bilateral_filter_floatmap(filt, img, sigmad, sigmar):
    depth_utils_cuda.bilateral_filter_floatmap(filt, img, sigmad, sigmar)

def median_fill_depthmap(filt, img, num_iters):
    assert(num_iters >= 2)
    for _ in range(num_iters//2):
        depth_utils_cuda.median_fill_depthmap(img, filt)
        depth_utils_cuda.median_fill_depthmap(filt, img)

def convert_depth_to_cameraspace(camspace, filt, intrinsic, depth_min, depth_max):
    depth_utils_cuda.convert_depth_to_cameraspace(camspace, filt, intrinsic, depth_min, depth_max)

def compute_normals(normals, camspace):
    depth_utils_cuda.compute_normals(normals, camspace)

class Depth2Normals(torch.nn.Module):
    def __init__(self, batch_size, width, height, depth_min, depth_max, max_num_fill_iters=40):
        super(Depth2Normals, self).__init__()
        self.width = width
        self.height = height
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.max_num_fill_iters = max_num_fill_iters
        # pre-allocatead helpers
        self.filter_helper = torch.zeros(batch_size, 1, height, width).cuda()
        self.camspace = torch.zeros(batch_size, height, width, 3).cuda()
        self.normals = torch.zeros(batch_size, height, width, 3).cuda()
        #self.return_depth = return_depth

    # debug access
    def get_campos(self):
        return self.camspace

    def forward(self, depth, intrinsic_params):
        bilateral_filter_floatmap(self.filter_helper, depth, 2.0, 0.1)
        if self.max_num_fill_iters > 0:
            invalid = (depth == 0).any()
            for _ in range(self.max_num_fill_iters//2):
                if not invalid:
                    break
                median_fill_depthmap(self.filter_helper, depth, 2)
                invalid = (depth == 0).any()
            if invalid:
                return None
        convert_depth_to_cameraspace(self.camspace, depth, intrinsic_params, self.depth_min, self.depth_max)
        compute_normals(self.normals, self.camspace)
        normals = self.normals.permute(0, 3, 1, 2).contiguous()
        #print('normals', normals.shape, self.normals.shape)
        return normals

