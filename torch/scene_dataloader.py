import os, sys, struct
import skimage.color as skcolor
import numpy as np
import torch
import torch.utils.data
import random
import math
import plyfile
#import time

import data_util
import utils.marching_cubes.marching_cubes as mc

def collate_voxels(batch):
    names = [x['name'] for x in batch]
    if None in names:
        return {'name': None, 'input': None, 'sdf': None}
    input = torch.stack([x['input'] for x in batch])
    mask = None
    if batch[0]['mask'] is not None:
        mask = torch.stack([x['mask'] for x in batch])
    known = None
    if batch[0]['known'] is not None:
        known = torch.stack([x['known'] for x in batch])
    colors = None
    if batch[0]['colors'] is not None:
        colors = torch.stack([x['colors'] for x in batch])
    images_depth = None
    images_color = None
    images_pose = None
    images_intrinsic = None    
    if np.all([x['image_color'] is not None for x in batch]):
        images_color = torch.cat([x['image_color'] for x in batch])
        images_pose = torch.cat([x['image_pose'] for x in batch])
        images_intrinsic = torch.cat([x['image_intrinsic'] for x in batch])
    if np.all([x['image_depth'] is not None for x in batch]):
        images_depth = torch.cat([x['image_depth'] for x in batch])
    sdfs = torch.stack([x['sdf'] for x in batch])
    world2grids = torch.stack([x['world2grid'] for x in batch])
    
    return {'name': names, 'input': input, 'sdf': sdfs, 'world2grid': world2grids, 'known': known, 'colors': colors, 'images_depth': images_depth, 'images_color': images_color, 'images_pose': images_pose, 'images_intrinsic': images_intrinsic, 'mask': mask}

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self, files, input_dim, truncation, load_known, augment_rgb_scaling, aug_scale_range, color_truncation, color_space, frame_info_path='', frame_path='', image_dims=(320,256), subsamp2d_factor=1, randomize_frames=True, load_depth=False, load_color=True, num_overfit=0, target_path='', max_input_height=0, scene_subsample_factor=1):
        self.is_chunks = target_path == '' # have target path -> full scene data
        if not target_path:
            self.files = [(f,f.replace('__inc__', '__cmp__')) for f in files if (os.path.isfile(f) and os.path.isfile(f.replace('__inc__', '__cmp__')))]
        else:
            self.files = [(f,os.path.join(target_path, os.path.basename(f))) for f in files if (os.path.isfile(f) and os.path.isfile(os.path.join(target_path, os.path.basename(f))))]
        self.input_dim = input_dim
        self.truncation = truncation
        self.color_truncation = color_truncation
        self.load_tgt_sparse = False
        self.load_known = load_known
        self.augment_rgb_scaling = augment_rgb_scaling
        self.aug_scale_range = aug_scale_range
        if self.augment_rgb_scaling:
            assert(self.aug_scale_range[1] > self.aug_scale_range[0])
            assert(self.aug_scale_range[0] >= 0)
        self.max_input_height = max_input_height
        self.scene_subsample_factor = scene_subsample_factor
        self.color_space = color_space
        self.frame_info_path = frame_info_path
        self.frame_path = frame_path
        self.image_dims = image_dims
        self.load_depth = load_depth
        self.load_color = load_color
        self.subsamp2d_factor = subsamp2d_factor
        self.randomize_frames = randomize_frames
        self.overfit = num_overfit > 0        
        self.idxs = None
        if self.overfit:
            num_repeat = max(1, num_overfit // len(self.files))
            self.files = self.files * num_repeat
            self.idxs = np.arange(len(self.files)).tolist()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):    
        inputsdf_file = self.files[idx][0]
        sdf_file = self.files[idx][1]        
        name = os.path.splitext(os.path.basename(inputsdf_file))[0]

        color_file = None if self.is_chunks else os.path.splitext(sdf_file)[0] + '.colors'
        sdf, world2grid, known, colors = data_util.load_sdf(sdf_file, load_sparse=self.load_tgt_sparse, load_known=self.load_known and self.is_chunks, load_colors=True, color_file=color_file)
        if sdf is None:
            return {'name': None}
        if self.load_known and not self.is_chunks:
            file_info = os.path.split(sdf_file)
            prefix = file_info[0] + '-complete' if 'color' in file_info[0] else file_info[0] + '_scanned'
            pad_known = (3,6,6) if 'color' in file_info[0] else (3,3,3)
            known_file = os.path.join(prefix, os.path.splitext(file_info[1])[0] + '.knw')
            known_file = known_file.replace('_trunc32-complete', '-complete')
            known = data_util.load_known(known_file, pad_known=pad_known, scale_to_dims=sdf.shape)
        input_color_file = None if self.is_chunks else os.path.splitext(inputsdf_file)[0] + '.colors'
        input, _, _, _, input_colors = data_util.load_sdf(inputsdf_file, load_sparse=True, load_known=False, load_colors=True, color_file=input_color_file)
        if input is None:
            return {'name': None}
        if self.color_truncation > 0:
            locs = input[0][np.abs(input[1]) > self.color_truncation] # to mask out
            input_colors[locs[:,0], locs[:,1], locs[:,2], :] = 0
        max_input_dim = np.max(input[0],0)
        if max_input_dim[0] >= sdf.shape[0] or max_input_dim[1] >= sdf.shape[1] or max_input_dim[2] >= sdf.shape[2]:
            mask = np.logical_and(input[0][:,0] < sdf.shape[0], np.logical_and(input[0][:,1] < sdf.shape[1], input[0][:,2] < sdf.shape[2]))
            input[0] = input[0][mask]
            input[1] = input[1][mask]

        image_depth = None
        image_color = None
        image_pose = None
        image_intrinsic = None
        if self.frame_info_path and self.frame_path:
            image_depth, image_color, image_pose, image_intrinsic, image_frameids = data_util.load_frames([name], [world2grid], self.frame_info_path, self.frame_path, randomize_frames=self.randomize_frames, depth_image_dims=self.image_dims, color_image_dims=self.image_dims, color_normalization=None, load_depth=self.load_depth, load_color=self.load_color)
                        
            if image_color is not None:
                if self.load_depth:
                    image_depth = image_depth.squeeze(1)
                if image_color is not None:
                    if len(image_color.shape) == 3:
                        image_color = image_color.squeeze(1)
                        image_pose = image_pose.squeeze(1)
                        image_intrinsic = image_intrinsic.squeeze(1)
                    else:
                        image_color = image_color.squeeze(0)
                        image_pose = image_pose.squeeze(0)
                        image_intrinsic = image_intrinsic.squeeze(0)
                if self.subsamp2d_factor > 1:
                    sz = image_color.shape[2:]
                    image_color = torch.nn.functional.interpolate(image_color, scale_factor=1.0/self.subsamp2d_factor, mode='bilinear', align_corners=False)
                    image_color = torch.nn.functional.interpolate(image_color, size=sz, mode='bilinear', align_corners=False)

        input_dense = data_util.sparse_to_dense_np(input[0], input[1][:,np.newaxis], sdf.shape[2], sdf.shape[1], sdf.shape[0], -float('inf'))
        if (self.is_chunks and (self.input_dim[0] != 96 and self.input_dim[0] != 128 and self.input_dim[0] != 160)) or self.scene_subsample_factor > 1:
            scale_factor = float(self.input_dim[0]) / 128 if self.is_chunks else 1.0/float(self.scene_subsample_factor)
            input_dense = torch.nn.functional.interpolate(torch.from_numpy(input_dense).unsqueeze(0).unsqueeze(0), scale_factor=scale_factor) * scale_factor
            input_dense = input_dense[0,0].numpy()
            input_colors = torch.nn.functional.interpolate(torch.from_numpy(input_colors).permute(3,0,1,2).contiguous().unsqueeze(0).float(), scale_factor=scale_factor)
            input_colors = input_colors[0].permute(1,2,3,0).contiguous().numpy().astype(np.uint8)
            sdf = torch.nn.functional.interpolate(torch.from_numpy(sdf).unsqueeze(0).unsqueeze(0), scale_factor=scale_factor) * scale_factor
            sdf = sdf[0,0].numpy()
            colors = torch.nn.functional.interpolate(torch.from_numpy(colors).permute(3,0,1,2).contiguous().unsqueeze(0).float(), scale_factor=scale_factor)
            colors = colors[0].permute(1,2,3,0).contiguous().numpy().astype(np.uint8)
            if known is not None:
                known = torch.nn.functional.interpolate(torch.from_numpy(known).float().unsqueeze(0).unsqueeze(0), scale_factor=scale_factor).byte()
                known = known[0,0].numpy()
            world2grid = np.matmul(data_util.make_scale_transform(scale_factor), world2grid).astype(np.float32)
        
        if self.augment_rgb_scaling:
            scale = np.random.rand(1) * (self.aug_scale_range[1] - self.aug_scale_range[0]) + self.aug_scale_range[0]
            
            input_colors = data_util.convert_rgbgrid_to_hsvgrid(input_colors.astype(np.float32)/255.0)
            colors = data_util.convert_rgbgrid_to_hsvgrid(colors.astype(np.float32)/255.0)
            # hue augmentation
            scaled = input_colors[:,:,:,0] * scale[0]
            cmask = scaled >= 360
            if np.sum(cmask) > 0:
                scaled[cmask] = scaled[cmask] % 360
            input_colors[:,:,:,0] = scaled
            input_colors = np.clip(data_util.convert_hsvgrid_to_rgbgrid(input_colors)*255, 0, 255).astype(np.uint8)
            scaled = colors[:,:,:,0] * scale[0]
            cmask = scaled >= 360
            if np.sum(cmask) > 0:
                scaled[cmask] = scaled[cmask] % 360
            colors[:,:,:,0] = scaled
            colors = np.clip(data_util.convert_hsvgrid_to_rgbgrid(colors)*255, 0, 255).astype(np.uint8)
            if image_color is not None:
                image_color = data_util.convert_rgbgrid_to_hsvgrid(image_color.permute(0,2,3,1).numpy())
                scaled = image_color[:,:,:,0] * scale[0]
                cmask = scaled >= 360
                if np.sum(cmask) > 0:
                    scaled[cmask] = scaled[cmask] % 360
                image_color[:,:,:,0] = scaled
                image_color = data_util.convert_hsvgrid_to_rgbgrid(image_color)
                image_color = torch.from_numpy(image_color).permute(0, 3, 1, 2).contiguous()
            
        if self.color_space == 'lab':
            colors = skcolor.rgb2lab(colors.astype(np.float32)/255).astype(np.float32)
            input_colors = skcolor.rgb2lab(input_colors.astype(np.float32)/255).astype(np.float32)
            # normalize tgt to 255 (expected in loss)
            colors[:,:,:,0] = (colors[:,:,:,0] / 100.0)*255.0
            colors[:,:,:,1:] = (colors[:,:,:,1:] + 100.0)/200.0*255.0
            # normalize input to 0/255
            input_colors[:,:,:,0] = input_colors[:,:,:,0] / 100.0
            input_colors[:,:,:,1:] = (input_colors[:,:,:,1:] + 100.0) / 200.0
            input_colors *= 255.0
            # images
            if image_color is not None:
                image_color = image_color.permute(0, 2, 3, 1).contiguous().view(1, -1, 3).numpy()
                image_color = skcolor.rgb2lab(image_color).astype(np.float32)
                image_color = np.transpose(image_color, [0,2,1]).reshape(1, 3, self.image_dims[1], self.image_dims[0])                
                # normalize to 0/1
                image_color[:,0] = image_color[:,0] / 100.0
                image_color[:,1:] = (image_color[:,1:] + 100.0)/200.0
                image_color = torch.from_numpy(image_color.astype(np.float32))

        empty = np.abs(input_dense) > self.truncation
        mask = np.zeros(input_dense.shape, dtype=np.float32)
        mask[input_dense <= -1] = 1
        mask[empty] = 0
        input_dense = data_util.preprocess_sdf_pt(input_dense, self.truncation)
        input_colors = input_colors.astype(np.float32) / 255.0
        input_colors[empty] = 0
        colors = torch.from_numpy(colors)
        input_dense = torch.from_numpy(input_dense).unsqueeze(0)
        input_colors = torch.from_numpy(input_colors).permute(3, 0, 1, 2).contiguous()
        input = torch.cat([input_dense, input_colors], 0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        sdf = sdf[np.newaxis,:]
        sdf = torch.from_numpy(sdf)
        world2grid = torch.from_numpy(world2grid)
        if self.load_known:
            known = known[np.newaxis,:]
            known = torch.from_numpy(known)
                
        sample = {'name': name, 'input': input, 'sdf': sdf, 'world2grid': world2grid, 'known': known, 'colors': colors, 'image_depth': image_depth, 'image_color': image_color, 'image_pose': image_pose, 'image_intrinsic': image_intrinsic, 'mask': mask}
        return sample



