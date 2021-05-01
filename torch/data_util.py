from __future__ import print_function

import os, sys, struct
import imageio
import numpy as np
import torch
import random
import math
import plyfile
from PIL import Image
import skimage.color as skcolor
import torchvision.transforms as transforms

import utils.marching_cubes.marching_cubes as mc
_SPLITTER = ','

def get_train_files(data_path, file_list, val_file_list, max_num):
    train_files, val_files = get_train_files_3d(data_path, file_list, val_file_list, max_num)
    _OVERFIT = len(train_files) == 1
    return train_files, val_files, _OVERFIT


def get_train_files_3d(data_path, file_list, val_file_list, max_num):
    names = open(file_list).read().splitlines()
    if not names[0].endswith('.sdf'):
        names = [name + '__0__.sdf' for name in names]
    files = [os.path.join(data_path, f) for f in names]
    val_files = []
    if val_file_list:
        val_names = open(val_file_list).read().splitlines()
        val_files = [os.path.join(data_path, f) for f in val_names]
    if max_num > 0:
        files = files[:max_num]
        val_files = val_files[:max_num]
    return files, val_files


def dump_args_txt(args, output_file):
    with open(output_file, 'w') as f:
        f.write('%s\n' % str(args))


# locs: zyx ordering
def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense #.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def dense_to_sparse_np(grid, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack(locs)
    return locs, values

def load_sdf(file, load_sparse, load_known, load_colors, is_sparse_file=True, color_file=None):
    #assert os.path.isfile(file)
    assert (not load_sparse and not load_known) or (load_sparse != load_known)
    try:
        fin = open(file, 'rb')
        dimx = struct.unpack('Q', fin.read(8))[0]
        dimy = struct.unpack('Q', fin.read(8))[0]
        dimz = struct.unpack('Q', fin.read(8))[0]
        voxelsize = struct.unpack('f', fin.read(4))[0]
        world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    except:
        print('failed to read file:', file)
        if load_sparse:
            return None, None, None, None, None
        else:
            return None, None, None, None
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    if is_sparse_file:
        num = struct.unpack('Q', fin.read(8))[0]
        locs = struct.unpack('I'*num*3, fin.read(num*3*4))
        locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
        locs = np.flip(locs,1).copy() # convert to zyx ordering
        sdf = struct.unpack('f'*num, fin.read(num*4))
        sdf = np.asarray(sdf, dtype=np.float32)
        sdf /= voxelsize
    else:
        raise # unimplemented
    known = None
    num_known = 0
    if load_colors and color_file is None: # chunk file
        num_known = struct.unpack('Q', fin.read(8))[0]
    if load_known or num_known > 0:
        if num_known != dimx * dimy * dimz:
            print('file', file)
            print('dims (%d, %d, %d) -> %d' % (dimx, dimy, dimz, dimx*dimy*dimz))
            print('#known', num_known)
            raw_input('sdlfkj')
        assert num_known == dimx * dimy * dimz
        known = struct.unpack('B'*num_known, fin.read(num_known))
        if load_known:
            known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
            mask = np.logical_and(sdf >= -1, sdf <= 1)
            known[locs[:,0][mask], locs[:,1][mask], locs[:,2][mask]] = 1
            mask = sdf > 1
            known[locs[:,0][mask], locs[:,1][mask], locs[:,2][mask]] = 0
        else:
            known = None
    colors = None
    if load_colors:
        if color_file is not None:
            with open(color_file, 'rb') as cfin:
                cdimx = struct.unpack('Q', cfin.read(8))[0]
                cdimy = struct.unpack('Q', cfin.read(8))[0]
                cdimz = struct.unpack('Q', cfin.read(8))[0]
                assert cdimx == dimx and cdimy == dimy and cdimz == dimz
                if is_sparse_file:
                    num = struct.unpack('Q', cfin.read(8))[0]
                    colors = struct.unpack('B'*num*3, cfin.read(num*3))
                    colors = np.asarray(colors, dtype=np.uint8).reshape(num, 3)
                    #TODO always loads dense
                    colors = sparse_to_dense_np(locs, colors, cdimx, cdimy, cdimz, 0)
                else:
                    colors = struct.unpack('B'*cdimz*cdimy*cdimx*3, cfin.read(cdimz*cdimy*cdimx*3))
                    colors = np.asarray(colors, dtype=np.uint8).reshape([cdimz, cdimy, cdimx, 3])
        else:
            num_color = struct.unpack('Q', fin.read(8))[0]
            assert num_color == dimx * dimy * dimz
            colors = struct.unpack('B'*num_color*3, fin.read(num_color*3))
            colors = np.asarray(colors, dtype=np.uint8).reshape([dimz, dimy, dimx, 3])
    fin.close()
    if load_sparse:
        return [locs, sdf], [dimz, dimy, dimx], world2grid, known, colors
    else:
        sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
        return sdf, world2grid, known, colors


#pad: padding at known scale
def load_known(file, pad_known, pad_scan=(3,6,6), scale_to_dims=None):
    #assert os.path.isfile(file)
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    known = struct.unpack('B'*dimz*dimy*dimx, fin.read(dimz*dimy*dimx))
    known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
    fin.close()
    if scale_to_dims is not None and known.shape != scale_to_dims:
        scale_factor = int(round(float(scale_to_dims[2] - 2*pad_scan[2]) / float(dimx - 2*pad_known[2])))
        pad_offset = np.array(pad_known, dtype=np.int32) * scale_factor - np.array(pad_scan, dtype=np.int32)
        known = known.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        known = known[pad_offset[0]:, pad_offset[1]:, pad_offset[2]:]
        known = known[:scale_to_dims[0], :scale_to_dims[1], :scale_to_dims[2]]
        assert(np.all(np.abs((known.shape[0] - scale_to_dims[0], known.shape[1] - scale_to_dims[1], known.shape[2] - scale_to_dims[2])) < 2))
        known = known[:scale_to_dims[0], :scale_to_dims[1], :scale_to_dims[2]]
    return known


def preprocess_sdf_pt(sdf, truncation):
    sdf[sdf < -truncation] = -truncation
    sdf[sdf > truncation] = truncation
    return sdf

def visualize_sdf_as_points(sdf, iso, output_file, transform=None):
    # collect verts from sdf
    verts = []
    for z in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for x in range(sdf.shape[2]):
                if abs(sdf[z,y,x]) < iso:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)

def visualize_colored_sdf_as_points(sdf, sdf_colors, iso, output_file, transform=None):
    # collect verts from sdf
    verts = []
    colors = []
    for z in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for x in range(sdf.shape[2]):
                if abs(sdf[z,y,x]) < iso:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
                    colors.append(np.array(sdf_colors[z, y, x]) / 255.0)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts)
    colors = np.stack(colors)
    visualize_points(verts, output_file, transform, colors)

def visualize_sparse_sdf_as_points(sdf_locs, sdf_vals, iso, output_file, transform=None, sdf_color=None):
    # collect verts from sdf
    mask = np.abs(sdf_vals) < iso
    verts = sdf_locs[:,:3][mask]
    colors = None
    if sdf_color is not None:
        colors = sdf_color[:,:3][mask]
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    if colors is not None:
        colors = np.stack(colors).astype(np.float32)
    visualize_points(verts, output_file, transform, colors)

def visualize_occ_as_points(sdf, thresh, output_file, color=None, transform=None, thresh_max = float('inf')):
    # collect verts from sdf
    verts = []
    colors = None if color is None else []
    for z in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for x in range(sdf.shape[2]):
                val = abs(sdf[z, y, x])
                if val > thresh and val < thresh_max:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
                    if color is not None:
                        colors.append(np.array(color[z,y,x,:])/255.0)
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    verts = np.stack(verts)
    if color is not None:
        colors = np.stack(colors)
    visualize_points(verts, output_file, transform, colors)
def visualize_colorgrid_as_points(colorgrid, output_file, thresh=15, transform=None):
    if np.max(colorgrid) < 1.1:
        print('[visualize_colorgrid_as_points] error: color range (', np.min(colorgrid), ',', np.max(colorgrid), ') expected [0,255]')
    # collect verts
    verts = []
    colors = []
    for z in range(colorgrid.shape[1]):
        for y in range(colorgrid.shape[2]):
            for x in range(colorgrid.shape[3]):
                c = colorgrid[:,z,y,x]
                if c[0] > thresh or c[1] > thresh or c[2] > thresh:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
                    colors.append(np.array(c) / 255.0)
                    
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    verts = np.stack(verts)
    colors = np.stack(colors)
    visualize_points(verts, output_file, transform, colors)

def visualize_sparse_locs_as_points(locs, output_file, transform=None, color=None):
    # collect verts from sdf
    verts = locs[:,:3]
    colors = None if color is None else color[:,:3]
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    if color is not None:
        colors = np.stack(colors).astype(np.float32) / 255.0
    visualize_points(verts, output_file, transform, colors)

def visualize_points(points, output_file, transform=None, colors=None):
    verts = points if points.shape[1] == 3 else np.transpose(points)
    if transform is not None:
        x = np.ones((verts.shape[0], 4))
        x[:, :3] = verts
        x = np.matmul(transform, np.transpose(x))
        x = np.transpose(x)
        verts = np.divide(x[:, :3], x[:, 3, None])

    ext = os.path.splitext(output_file)[1]
    if colors is not None or ext == '.obj':
        output_file = os.path.splitext(output_file)[0] + '.obj'
        num_verts = len(verts)
        with open(output_file, 'w') as f:
            for i in range(num_verts):
                v = verts[i]
                if colors is None:
                    f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                else:
                    f.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], colors[i,0], colors[i,1], colors[i,2]))
    elif ext == '.ply':
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = plyfile.PlyElement.describe(verts,'vertex')
        plyfile.PlyData([el]).write(output_file)
    else:
        raise # unsupported extension

def make_scale_transform(scale):
    if isinstance(scale, int) or isinstance(scale, float):
        scale = [scale, scale, scale]
    assert( len(scale) == 3 )
    transform = np.eye(4, 4)
    for k in range(3):
        transform[k,k] = scale[k]
    return transform


def convert_hsv_to_rgb(hsv):
    H = hsv[0]
    S = hsv[1]
    V = hsv[2]

    hd = H / 60.0
    h = int(hd)
    f = hd - h

    p = V*(1.0 - S)
    q = V*(1.0 - S*f)
    t = V*(1.0 - S*(1.0 - f))
    
    if (h == 0 or h == 6):
        return np.array([V, t, p])
    elif (h == 1):
        return np.array([q, V, p])
    elif (h == 2):
        return np.array([p, V, t])
    elif (h == 3):
        return np.array([p, q, V])
    elif (h == 4):
        return np.array([t, p, V])
    else:
        return np.array([V, p, q])

def convert_rgbgrid_to_hsvgrid(rgb):
    assert(len(rgb.shape) == 4 and isinstance(rgb[0,0,0,0], np.float32))
    
    mn = np.min(rgb, 3)
    mx = np.max(rgb, 3)
    nonzero_mask = np.sum(rgb,3) > 0.01
    
    hsv = np.zeros_like(rgb)
    # v
    hsv[:,:,:,2] = mx
    delta = mx - mn;
    # s
    hsv[nonzero_mask,1] = delta[nonzero_mask]/mx[nonzero_mask]
    # h
    mask = delta == 0
    hsv[mask,0] = 0
    mask = np.logical_not(mask)
    mask2 = np.logical_and(mask, rgb[:,:,:,0] == mx)
    hsv[mask2,0] = 60.0 * (0 + (rgb[mask2,1] - rgb[mask2,2]) / delta[mask2])
    mask2 = np.logical_and(mask, rgb[:,:,:,1] == mx)
    hsv[mask2,0] = 60.0 * (2 + (rgb[mask2,2] - rgb[mask2,0]) / delta[mask2])
    mask2 = np.logical_and(mask, rgb[:,:,:,2] == mx)
    hsv[mask2,0] = 60.0 * (4 + (rgb[mask2,0] - rgb[mask2,1]) / delta[mask2])
    mask2 = hsv[:,:,:,0] < 0
    if np.sum(mask2) > 0:
        hsv[mask2,0] += 360
    return hsv

def vis_depth_as_hsv(depths, depth_max):
    sz = depths.shape
    assert(len(sz) == 3 or len(sz) == 4)
    depths = depths.flatten()
    depths[depths == -float('inf')] = 0
    depths = 359*np.clip(depths/depth_max, 0, 1)
    depths = np.stack([depths, np.ones(depths.shape[0]), np.ones(depths.shape[0])*0.5], 1)
    depths = convert_hsvgrid_to_rgbgrid(depths)
    if len(sz) == 3:
        depths = depths.reshape(sz[0], sz[1], sz[2], 3)
    elif len(sz) == 4:
        depths = depths.reshape(sz[0], sz[1], sz[2], sz[3], 3)
    return depths
    
def convert_hsvgrid_to_rgbgrid(hsv):
    sz = hsv.shape
    assert(len(sz) == 2 or len(sz) == 4 or len(sz) == 5)
    if len(sz) == 2:
        hd = hsv[:,0] / 60.0
    elif len(sz) == 4:
        hd = hsv[:,:,:,0] / 60.0
    else:
        hd = hsv[:,:,:,:,0] / 60.0
    h = np.floor(hd)
    f = hd - h
    
    if len(sz) == 2:
        p = hsv[:,2]*(1.0 - hsv[:,1])
        q = hsv[:,2]*(1.0 - hsv[:,1]*f)
        t = hsv[:,2]*(1.0 - hsv[:,1]*(1.0-f))
    elif len(sz) == 4:
        p = hsv[:,:,:,2]*(1.0 - hsv[:,:,:,1])
        q = hsv[:,:,:,2]*(1.0 - hsv[:,:,:,1]*f)
        t = hsv[:,:,:,2]*(1.0 - hsv[:,:,:,1]*(1.0-f))
    else:
        p = hsv[:,:,:,:,2]*(1.0 - hsv[:,:,:,:,1])
        q = hsv[:,:,:,:,2]*(1.0 - hsv[:,:,:,:,1]*f)
        t = hsv[:,:,:,:,2]*(1.0 - hsv[:,:,:,:,1]*(1.0-f))
    
    rgb = np.zeros_like(hsv)
    mask = np.logical_or(h == 0, h == 6)
    rgb[mask,0] = hsv[mask,2]
    rgb[mask,1] = t[mask]
    rgb[mask,2] = p[mask]
    mask = (h == 1)
    rgb[mask,0] = q[mask]
    rgb[mask,1] = hsv[mask,2]
    rgb[mask,2] = p[mask]
    mask = (h == 2)
    rgb[mask,0] = p[mask]
    rgb[mask,1] = hsv[mask,2]
    rgb[mask,2] = t[mask]
    mask = (h == 3)
    rgb[mask,0] = p[mask]
    rgb[mask,1] = q[mask]
    rgb[mask,2] = hsv[mask,2]
    mask = (h == 4)
    rgb[mask,0] = t[mask]
    rgb[mask,1] = p[mask]
    rgb[mask,2] = hsv[mask,2]
    mask = (h == 5)
    rgb[mask,0] = hsv[mask,2]
    rgb[mask,1] = p[mask]
    rgb[mask,2] = q[mask]
    return rgb

def convert_lab_to_rgb(colorgrid):
    sz = colorgrid.shape
    assert(len(sz) == 2 or len(sz) == 3 or len(sz) == 5)
    if len(sz) == 2:
        colorgrid[:,0] *= 100.0
        colorgrid[:,1:] = (colorgrid[:,1:] * 2.0 - 1.0) * 100.0
        colorgrid = colorgrid[np.newaxis,:]
    else:
        if len(sz) == 5:
            colorgrid = colorgrid.reshape(sz[0], sz[1]*sz[2]*sz[3], 3)
        colorgrid[:,:,0] *= 100.0
        colorgrid[:,:,1:] = (colorgrid[:,:,1:] * 2.0 - 1.0) * 100.0
    colors = skcolor.lab2rgb(colorgrid)
    if len(sz) == 2:
        colors = colors[0]
    elif len(sz) == 5:
        colors = colors.reshape(sz[0], sz[1], sz[2], sz[3], sz[4])
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    return colors

def convert_rgb_to_lab_pt(colorgrid):
    sz = colorgrid.shape
    if sz[1] == 3:
        colorgrid = colorgrid.permute(0,2,3,1).contiguous()
        sz = colorgrid.shape
    # convert to xyz
    mask = colorgrid > 0.0405
    colorgrid[mask] = torch.pow((colorgrid[mask] + 0.055) / 1.055, 2.4)
    colorgrid[~mask] /= 12.92
    
    xyz_from_rgb = torch.Tensor([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]]).to(colorgrid.device)
    colorgrid = colorgrid.view(-1, 3).t()
    colorgrid = torch.matmul(xyz_from_rgb, colorgrid)
    colorgrid = colorgrid.t()
    
    colorgrid[:,0] /= 0.95047
    #colorgrid[:,1] /= 1
    colorgrid[:,2] /= 1.08883
    mask = colorgrid > 0.008856
    colorgrid[mask] = torch.pow(colorgrid[mask], 1.0/3.0)
    colorgrid[~mask] = 7.787 * colorgrid[~mask] + 16.0 / 116.0
    colors = torch.stack([116.0*colorgrid[:,1]-16.0, 500.0*(colorgrid[:,0]-colorgrid[:,1]), 200.0*(colorgrid[:,1]-colorgrid[:,2])], 1)
    return colors.view(sz)
def convert_lab_to_rgb_pt(colorgrid):
    colorgrid = colorgrid.view(-1,3)
    # convert to xyz
    y = (colorgrid[:,0] + 16.0) / 116.0
    x = (colorgrid[:,1] / 500.0) + y
    z = y - (colorgrid[:,2] / 200.0)
    #print('z', torch.sum(z<0).item())
    z[z < 0] = 0 # invalid
    xyz = torch.stack([x,y,z],1)
    mask = xyz > 0.2068966
    m1 = torch.pow(xyz[mask], 3.0)
    m0 = (xyz[~mask] - 16.0 / 116.0) / 7.787
    xyz = xyz.masked_scatter(mask, m1)
    xyz = xyz.masked_scatter(~mask, m0)
    x = xyz[:,0] * 0.95047
    y = xyz[:,1] #*1
    z = xyz[:,2] * 1.08883
    xyz = torch.stack([x,y,z],1)  
    rgb_from_xyz = torch.Tensor([[ 3.2405, -1.5372, -0.4985],
        [-0.9693,  1.8760,  0.0416],
        [ 0.0556, -0.2040,  1.0573]]).to(colorgrid.device)
    rgb = torch.matmul(rgb_from_xyz, xyz.t()).t()
    mask = rgb > 0.0031308
    m1 = 1.055 * torch.pow(rgb[mask], 1.0 / 2.4) - 0.055
    m0 = rgb[~mask] * 12.92
    rgb = rgb.masked_scatter(mask, m1)
    rgb = rgb.masked_scatter(~mask, m0)
    rgb = torch.clamp(rgb, 0, 1)
    return rgb
    #mask = xyz > 0.2068966
    #xyz[mask] = torch.pow(xyz[mask], 3.0)
    #xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787    
    #xyz[:,0] *= 0.95047
    ##xyz[:,1] *= 1
    #xyz[:,2] *= 1.08883    
    #rgb_from_xyz = torch.Tensor([[ 3.2405, -1.5372, -0.4985],
    #    [-0.9693,  1.8760,  0.0416],
    #    [ 0.0556, -0.2040,  1.0573]]).to(colorgrid.device)
    #rgb = torch.matmul(rgb_from_xyz, xyz.t()).t()
    #mask = rgb > 0.0031308
    #rgb[mask] = 1.055 * torch.pow(rgb[mask], 1.0 / 2.4) - 0.055
    #rgb[~mask] *= 12.92
    #rgb = torch.clamp(rgb, 0, 1)
    #return rgb
def convert_lab01_to_rgb_pt(colorgrid):
    sz = colorgrid.shape
    colorgrid = colorgrid.view(-1,3)
    l = colorgrid[:,:1]
    ab = colorgrid[:,1:]
    l = 100.0*l
    ab = (ab * 2.0 - 1.0) * 100.0
    colorgrid = torch.cat([l,ab],1)
    #colorgrid[:,0] = 100.0*colorgrid[:,0]
    #colorgrid[:,1:] = (colorgrid[:,1:] * 2.0 - 1.0) * 100.0
    return convert_lab_to_rgb_pt(colorgrid).view(sz)


def save_predictions(output_path, indices, names, inputs, target_for_sdf, target_for_colors, target_hier, target_images, output_sdf, output_color, output_hier, output_images, world2grids, truncation, color_space='rgb', input_images=None, pred_depth=None, target_depth=None, pred_occ=None, thresh=1, pred_color=False, aux_images=None):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if target_for_sdf is not None:
        dims = target_for_sdf.shape[2:]
    elif inputs is not None:
        dims = inputs.shape[2:]
    isovalue = 0
    trunc = truncation - 0.1
    ext = '.ply'
    pred_color = pred_color or (output_color[0] is not None)
    
    input_color_images = None
    target_color_images = None
    output_color_images = None
    input_normal_images = None
    target_normal_images = None
    output_normal_images = None
    if target_images is not None:
        if pred_color:
            target_color_images = target_images[:,:,:,:3]
        if target_images.shape[3] > 3 or not pred_color:
            target_normal_images = target_images[:,:,:,-3:]
    if output_images is not None:
        if pred_color:
            output_color_images = output_images[:,:,:,:3]
        if output_images.shape[3] > 3 or not pred_color:
            output_normal_images = output_images[:,:,:,-3:]
    if input_images is not None:
        if pred_color:
            input_color_images = input_images[:,:,:,:3]
        if input_images.shape[3] > 3 or not pred_color:
            input_normal_images = input_images[:,:,:,-3:]
    if aux_images is not None and pred_color:
        aux_images = aux_images[:,:,:,:3]
    
    if color_space == 'lab':
        if output_color is not None:
            for k in range(len(output_color)):
                if output_color[k] is not None:
                    convert = convert_lab_to_rgb(output_color[k].astype(np.float32)/255.0).astype(np.float32)
                    convert = np.clip(1.368 * (convert - 128) + 128, 0, 255)
                    output_color[k] = convert.astype(np.uint8)
        if inputs is not None and inputs.shape[1] > 1:
            colors = inputs[:,1:]
            colors = convert_lab_to_rgb(colors.transpose(0, 2, 3, 4, 1)).astype(np.float32)/255.0
            inputs[:,1:] = colors.transpose(0, 4, 1, 2, 3)
        if target_for_colors is not None:
            target_for_colors = convert_lab_to_rgb(target_for_colors.astype(np.float32)/255.0)
        if target_color_images is not None:
            for k in range(len(target_color_images)):
                if target_color_images[k] is not None:
                    target_color_images[k] = convert_lab_to_rgb(target_color_images[k][np.newaxis,np.newaxis,:])[0,0]
                if output_color_images[k] is not None:
                    output_color_images[k] = convert_lab_to_rgb(output_color_images[k][np.newaxis,np.newaxis,:])[0,0].astype(np.uint8)
            target_color_images = target_color_images.astype(np.uint8)
            output_color_images = output_color_images.astype(np.uint8)
        if input_color_images is not None:
            for k in range(len(input_color_images)):
                if input_color_images[k] is not None:
                    input_color_images[k] = convert_lab_to_rgb(input_color_images[k][np.newaxis,np.newaxis,:])[0,0]
            input_color_images = input_color_images.astype(np.uint8)
        if aux_images is not None:
            for k in range(len(aux_images)):
                if aux_images[k] is not None:
                    aux_images[k] = convert_lab_to_rgb(aux_images[k][np.newaxis,np.newaxis,:])[0,0]
            aux_images = aux_images.astype(np.uint8)
    else:
        if target_color_images is not None:
            target_color_images = np.clip(255*target_color_images, 0, 255).astype(np.uint8)
        if output_color_images is not None:
            output_color_images = np.clip(255*output_color_images, 0, 255).astype(np.uint8)
        if input_color_images is not None:
            input_color_images = np.clip(255*input_color_images, 0, 255).astype(np.uint8)
        if aux_images is not None:
            aux_images = np.clip(255*aux_images, 0, 255).astype(np.uint8)

    for k in range(len(indices)):
        name = names[indices[k]].replace('/', '_')
        if inputs is not None:
            input_sdf = inputs[k,0]
            colors = None
            if inputs.shape[1] > 1:
                colors = (inputs[k,1:4] * 255).transpose(1, 2, 3, 0)
                colors = torch.from_numpy(colors.astype(np.uint8)).contiguous()            
            mc.marching_cubes(torch.from_numpy(input_sdf), colors.byte(), isovalue=isovalue, truncation=trunc, thresh=10, output_filename=os.path.join(output_path, name + 'input-mesh' + ext))
        if output_sdf[k] is not None:
            locs = output_sdf[k][0][:,:3]
            pred_sdf_dense = sparse_to_dense_np(locs, output_sdf[k][1][:,np.newaxis], dims[2], dims[1], dims[0], -float('inf'))
            colors = None
            if output_color is not None and output_color[k] is not None:
                colors = sparse_to_dense_np(locs, output_color[k], dims[2], dims[1], dims[0], 0)
                colors = torch.from_numpy(colors)
                colors = colors.byte()
            if pred_sdf_dense is not None:
                mc.marching_cubes(torch.from_numpy(pred_sdf_dense), colors, isovalue=isovalue, truncation=trunc, thresh=10, output_filename=os.path.join(output_path, name + 'pred-mesh' + ext))
        if target_for_sdf is not None:
            target = target_for_sdf[k,0]
            colors = None
            if target_for_colors is not None:
                colors = torch.from_numpy(target_for_colors[k])
                colors = colors.byte()
            mc.marching_cubes(torch.from_numpy(target), colors, isovalue=isovalue, truncation=trunc, thresh=10, output_filename=os.path.join(output_path, name + 'target-mesh' + ext))
        if target_color_images is not None and target_color_images[k] is not None:        
            Image.fromarray(target_color_images[k]).save(os.path.join(output_path, name + '_target.png'))
        if output_color_images is not None and output_color_images[k] is not None:
            Image.fromarray(output_color_images[k]).save(os.path.join(output_path, name + '_pred.png'))
        if input_color_images is not None and input_color_images[k] is not None:
            Image.fromarray(input_color_images[k]).save(os.path.join(output_path, name + '_input.png'))
        if aux_images is not None and aux_images[k] is not None:
            Image.fromarray(aux_images[k]).save(os.path.join(output_path, name + '_aux.png'))
        if target_normal_images is not None:
            vis = np.clip((target_normal_images[k]+1)*0.5*255, 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_path, name + '_target-normals.png'))
        if output_normal_images is not None:
            vis = np.clip((output_normal_images[k]+1)*0.5*255, 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_path, name + '_pred-normals.png'))
        if input_normal_images is not None:
            vis = np.clip((input_normal_images[k]+1)*0.5*255, 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_path, name + '_input-normals.png'))
        if pred_depth is not None:
            vis = np.clip(pred_depth[k]*255, 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_path, name + '_pred-depth.png'))
        if target_depth is not None:
            vis = np.clip(target_depth[k]*255, 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_path, name + '_target-depth.png'))
        
        if target_color_images is not None and target_color_images[k] is None and target_for_colors is not None and target_for_colors.shape[1] == 1:        
            # vis input/pred/tgt images
            input = np.clip(255*sparse_to_dense_np(inputs[0][inputs[0][:,-1] == k][:,:-1], feats[:,1:], dims[2], dims[1], dims[0], -float('inf')), 0, 255).astype(np.uint8)[0]
            target = np.clip(target_for_colors[k], 0, 255).astype(np.uint8)[0]
            pred = np.clip(sparse_to_dense_np(output_sdf[k][0][:,:3], output_color[k], dims[2], dims[1], dims[0], 0), 0, 255).astype(np.uint8)[0]
            Image.fromarray(input).save(os.path.join(output_path, name + '_input.png'))
            Image.fromarray(target).save(os.path.join(output_path, name + '_target.png'))
            Image.fromarray(pred).save(os.path.join(output_path, name + '_pred.png'))


def read_frame_file(filename, randomize, max_num_frames):
    lines = open(filename).read().splitlines()
    frames = [int(line) for line in lines]
    if randomize:
        frames = np.array(frames)
        frames = frames[frames != -1]
        random.shuffle(frames)
    return frames[:max_num_frames]

def read_camera_file(filename, intrinsic_filename):
    lines = open(filename).read().splitlines()
    #assert len(lines) == 8
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    lines = np.asarray(lines, dtype=np.float32)
    pose = torch.from_numpy(lines[:4])
    if intrinsic_filename is None:
        intrinsic = torch.from_numpy(lines[4:])
    else:
        lines = open(intrinsic_filename).read().splitlines()
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
        lines = np.asarray(lines, dtype=np.float32)
        intrinsic = torch.from_numpy(lines[:4])
    return pose, intrinsic    

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    if image.size[0] == new_image_dims[1] and image.size[1] == new_image_dims[0]:
        return image
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0,0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1,1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0,2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic

# assumes color_image is 3xhxw and [0,1] float
def vis_point_cloud(depth_image, color_image, intrinsic, extrinsic, output_file):
    intrinsic_inv = np.linalg.inv(intrinsic)
    verts = []
    colors = []
    for r in range(depth_image.shape[0]):
        for c in range(depth_image.shape[1]):
            d = depth_image[r,c]
            if d != 0:
                campos = np.matmul(intrinsic_inv, np.array([float(c)*d, float(r)*d, d, 0.0]))[:3]
                verts.append(campos)
                color = color_image[:,r,c]
                colors.append(color)
    if len(verts) == 0:
        print('warning: no valid points for %s' % output_file)
        return
    verts = np.stack(verts)
    colors = np.stack(colors)
    visualize_points(verts, output_file, extrinsic, colors)

# depth_image_dims: width, height
def load_frame(depth_file, color_file, camera_file, depth_image_dims, color_image_dims, normalize, load_depth, load_color, intrinsic_file=None):
    pose, intrinsic = read_camera_file(camera_file, intrinsic_file)
    depth_image = None
    color_image = None
    orig_dims = None
    if load_depth:
        depth_image = imageio.imread(depth_file)
        orig_dims = [depth_image.shape[1], depth_image.shape[0]]
        depth_image = resize_crop_image(depth_image, depth_image_dims)
        depth_image = depth_image.astype(np.float32) / 1000.0
        depth_image = torch.from_numpy(depth_image)
    if load_color:
        color_image = imageio.imread(color_file)
        orig_dims = [color_image.shape[1], color_image.shape[0]]
        color_image = resize_crop_image(color_image, color_image_dims)
        color_image = np.transpose(color_image, [2, 0, 1])  # move feature to front
        color_image = torch.Tensor(color_image.astype(np.float32) / 255.0)
        if normalize is not None:
            color_image = normalize(color_image)
    if color_image_dims != orig_dims:
        intrinsic = adjust_intrinsic(intrinsic, orig_dims, color_image_dims)
    return depth_image, color_image, pose, intrinsic

def load_frames(names, world2grids, frame_path, image_path, randomize_frames, depth_image_dims, color_image_dims, color_normalization, load_depth, load_color, max_num_frames=1):
    batch_size = len(names)
    poses = torch.FloatTensor(batch_size, max_num_frames, 4, 4)
    intrinsics = torch.FloatTensor(batch_size, max_num_frames, 4)
    scenes = [name.split('_room')[0] for name in names]
    frames = []
    if frame_path == 'self':
        frames = [[int(name.split('__inc__')[1])] for name in names]
    else:
        frame_files = [os.path.join(frame_path, name.replace('__inc__', '__cmp__') + '.txt') for name in names]
        frames = [read_frame_file(frame_file,randomize_frames,max_num_frames) for frame_file in frame_files]    
    if len(frames[0]) < max_num_frames:
        #print('not enough frames', names[0])
        return None, None, None, None, None
    depth_files = [[os.path.join(image_path, scene + '/depth/' + str(f) + '.png') for f in frameids] for frameids,scene in zip(frames,scenes)]
    color_files = [[os.path.join(image_path, scene + '/color/' + str(f) + '.jpg') for f in frameids] for frameids,scene in zip(frames,scenes)]
    camera_files = [[os.path.join(image_path, scene + '/camera/' + str(f) + '.txt') for f in frameids] for frameids,scene in zip(frames,scenes)]    
    depths = None if not load_depth else torch.zeros(batch_size, max_num_frames, depth_image_dims[1], depth_image_dims[0], dtype=torch.float)
    colors = None if not load_color else torch.zeros(batch_size, max_num_frames, 3, color_image_dims[1], color_image_dims[0], dtype=torch.float)
    for b in range(batch_size):
        for f in range(max_num_frames):
            depth_image, color_image, pose, intrinsic = load_frame(depth_files[b][f], color_files[b][f], camera_files[b][f], depth_image_dims, color_image_dims, color_normalization, load_depth, load_color)
            if load_depth:
                depths[b,f] = depth_image
            if load_color:
                colors[b,f] = color_image
            poses[b,f] = pose
            intrinsics[b,f,0] = intrinsic[0,0]
            intrinsics[b,f,1] = intrinsic[1,1]
            intrinsics[b,f,2] = intrinsic[0,2]
            intrinsics[b,f,3] = intrinsic[1,2]
    return depths, colors, poses, intrinsics, frames

# images: b x 3 x h x w
def compute_crops(images):
    crops = []
    extent_c = 0
    extent_r = 0
    for b in range(len(images)):
        mask = images[b,0] != -float('inf')
        nz = torch.nonzero(mask)
        if len(nz) == 0:
            cmin = 0
            cmax = 0
            rmin = 0
            rmax = 0
        else:
            cmin = torch.min(nz[:,0]).item()
            cmax = torch.max(nz[:,0]).item()
            rmin = torch.min(nz[:,1]).item()
            rmax = torch.max(nz[:,1]).item()
        extent_c = max(100, max(cmax - cmin, extent_c))
        extent_r = max(100, max(rmax - rmin, extent_r))
        crops.append([cmin,cmax,rmin,rmax])
    for k in range(len(crops)):
        cmin = max(0, crops[k][1]-extent_c)
        cmax = cmin + extent_c
        rmin = max(0, crops[k][3]-extent_r)
        rmax = rmin + extent_r
        crops[k] = [cmin,cmax,rmin,rmax]
    return crops

    
    


