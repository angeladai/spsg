from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import shutil
import random
import torch
import numpy as np
import gc

import data_util
import scene_dataloader
import model as model_util
import loss as loss_util

COLOR_SPACES = ['rgb', 'lab']

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', required=True, help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
# model params
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=0, help='#points / voxel dim.')
parser.add_argument('--nf_gen', type=int, default=20, help='controls #channels of generator')
parser.add_argument('--no_pass_geo_feats', dest='pass_geo_feats', action='store_false')
parser.add_argument('--input_mask', type=int, default=1, help='input mask')
# test params
parser.add_argument('--max_input_height', type=int, default=128, help='truncate input to this height (in voxels), 0 to disable')
parser.add_argument('--num_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--max_to_process', type=int, default=150, help='max num to process')
parser.add_argument('--vis_only', dest='vis_only', action='store_true')
parser.add_argument('--weight_color_loss', type=float, default=1.0, help='weight color loss vs rest (0 to disable).')
parser.add_argument('--color_thresh', type=float, default=15.0, help='mask colors with all values < color_thresh')
parser.add_argument('--color_truncation', type=float, default=0, help='truncation in voxels for color')
parser.add_argument('--augment_rgb_scaling', dest='augment_rgb_scaling', action='store_true')
parser.add_argument('--augment_scale_min', type=float, default=0.5, help='for color augmentation')
parser.add_argument('--augment_scale_max', type=float, default=1.5, help='for color augmentation')
parser.add_argument('--color_space', type=str, default='lab', help='[rgb, lab]')
parser.add_argument('--cpu', dest='cpu', action='store_true')


parser.set_defaults(vis_only=False, augment_rgb_scaling=False, cpu=False, pass_geo_feats=True)
args = parser.parse_args()
if args.input_dim == 0: # set default values
    args.input_dim = (128, 64, 64)
args.input_nf = 4
UP_AXIS = 0 # z is 0th 
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# create model
nf_in_color = 3 if args.input_mask == 0 else 4
model = model_util.Generator(nf_in_geo=1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats, truncation=args.truncation, max_data_size=args.input_dim)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)
if not args.cpu:
    model = model.cuda()


def test(dataloader, output_vis, num_to_vis):
    model.eval()

    hierarchy_factor = 4
    num_proc = 0
    num_vis = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):            
            inputs = sample['input']
            mask = sample['mask']
            max_input_dim = np.array(inputs.shape[2:])
            if args.max_input_height > 0 and max_input_dim[UP_AXIS] > args.max_input_height:
                max_input_dim[UP_AXIS] = args.max_input_height
                mask_input = inputs[0][:,UP_AXIS] < args.max_input_height
                inputs = inputs[:,:,:args.max_input_height]
                if mask is not None:
                    mask = mask[:,:,:args.max_input_height]
            max_input_dim = ((max_input_dim + hierarchy_factor - 1) // hierarchy_factor) * hierarchy_factor
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (num_proc, args.max_to_process, sample['name'], max_input_dim[0], max_input_dim[1], max_input_dim[2]))
            # pad target to max_input_dim
            padded = torch.zeros(1, inputs.shape[1], max_input_dim[0], max_input_dim[1], max_input_dim[2])
            padded[:,0].fill_(-args.truncation)
            padded[:, :, :min(args.max_input_height,inputs.shape[2]), :inputs.shape[3], :inputs.shape[4]] = inputs[:, :, :args.max_input_height, :, :]
            inputs = padded
            padded_mask = torch.zeros(1, 1, max_input_dim[0], max_input_dim[1], max_input_dim[2])
            padded_mask[:, :, :min(args.max_input_height,mask.shape[2]), :mask.shape[3], :mask.shape[4]] = mask[:, :, :args.max_input_height, :, :]
            mask = padded_mask

            model.update_sizes(max_input_dim)  
            output_occ = None          
            try:
                if not args.cpu:
                    inputs = inputs.cuda()
                    mask = mask.cuda()
                output_occ, output_sdf, output_color = model(inputs, mask, pred_sdf=[True, True], pred_color=args.weight_color_loss > 0)
                
                if output_occ is not None:
                    occ = torch.nn.Sigmoid()(output_occ.detach()) > 0.5
                    locs = torch.nonzero((torch.abs(output_sdf.detach()[:,0]) < args.truncation) & occ[:,0]).cpu()
                else:
                    locs = torch.nonzero(torch.abs(output_sdf[:,0]) < args.truncation).cpu()
                locs = torch.cat([locs[:,1:], locs[:,:1]],1)
                output_sdf = [locs, output_sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].detach().cpu()]
                if args.weight_color_loss == 0:
                    output_color = None
                else:
                    output_color = [locs, output_color[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]]]
                if output_color is not None:
                    output_color = (output_color[1] + 1) * 0.5
            except:
                print('exception')
                gc.collect()
                continue

            num_proc += 1
            if num_vis < num_to_vis:
                vis_pred_sdf = [None]
                vis_pred_color = [None]
                if len(output_sdf[0]) > 0:
                    if output_color is not None: # convert colors to vec3uc
                        output_color = torch.clamp(output_color.detach() * 255, 0, 255)
                    vis_pred_sdf[0] = [output_sdf[0].cpu().numpy(), output_sdf[1].squeeze().cpu().numpy()]
                    vis_pred_color[0] = output_color.cpu().numpy()
                vis_pred_images_color = None
                vis_tgt_images_color = None
                data_util.save_predictions(output_vis, np.arange(1), sample['name'], inputs.cpu().numpy(), None, None, None, vis_tgt_images_color, vis_pred_sdf, vis_pred_color, None, vis_pred_images_color, sample['world2grid'], args.truncation, args.color_space)
                num_vis += 1
            gc.collect()
    sys.stdout.write('\n')


def main():
    # data files
    test_files, _, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '', 0)
    if len(test_files) > args.max_to_process:
        test_files = test_files[:args.max_to_process]
    else:
        args.max_to_process = len(test_files)
    random.seed(42)
    random.shuffle(test_files)
    print('#test files = ', len(test_files))
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, True, args.augment_rgb_scaling, (args.augment_scale_min, args.augment_scale_max), args.color_truncation, args.color_space, target_path=args.target_data_path, max_input_height=args.max_input_height)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate_voxels)

    if os.path.exists(args.output):
        if args.vis_only:
            print('warning: output dir %s exists, will overwrite any existing files')
        else:
            input('warning: output dir %s exists, press key to delete and continue' % args.output)
            shutil.rmtree(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    output_vis_path = os.path.join(args.output, 'vis')
    if not os.path.exists(output_vis_path):
        os.makedirs(output_vis_path)

    # start testing
    print('starting testing...')
    test(test_dataloader, output_vis_path, args.num_to_vis)




if __name__ == '__main__':
    main()


