#import os
#import numpy as np
import torch

import color_utils_cpp

# colors: rgb at last channel, 0-1 range
# hsv output in 360,0-1
def convert_rgb_to_hsv(rgb, normalized01):
    sz = rgb.shape
    rgb = rgb.view(-1, 3)
    hsv = torch.zeros_like(rgb)
    color_utils_cpp.convert_rgb_to_hsv(rgb, hsv, normalized01)
    hsv = hsv.view(sz)
    return hsv

# hsv in 360,0-1, rgb in 0-1
def convert_hsv_to_rgb(hsv, normalized01):
    sz = hsv.shape
    hsv = hsv.view(-1, 3)
    rgb = torch.zeros_like(hsv)
    color_utils_cpp.convert_hsv_to_rgb(hsv, rgb, normalized01)
    rgb = torch.clamp(rgb, 0, 1).view(sz)
    return rgb
    
# rgb in 0-1, lab in 0-100,-100/100
def convert_rgb_to_lab(rgb, normalized01):
    sz = rgb.shape
    rgb = rgb.view(-1, 3)
    lab = torch.zeros_like(rgb)
    color_utils_cpp.convert_rgb_to_lab(rgb, lab, normalized01)
    lab = lab.view(sz)
    return lab

# lab in 0-100,-100/100, rgb in 0-1
def convert_lab_to_rgb(lab, normalized01):
    sz = lab.shape
    lab = lab.view(-1, 3)
    rgb = torch.zeros_like(lab)
    color_utils_cpp.convert_lab_to_rgb(lab, rgb, normalized01)
    rgb = torch.clamp(rgb, 0, 1).view(sz)
    return rgb