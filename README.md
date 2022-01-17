# SPSG

SPSG presents a self-supervised approach to generate high-quality, colored 3D models of scenes from RGB-D scan observations by learning to infer unobserved scene geometry and color. Rather than relying on 3D reconstruction losses to inform our 3D geometry and color reconstruction, we propose adversarial and perceptual losses operating on 2D renderings in order to achieve high-resolution, high-quality colored reconstructions of scenes.  For more details, please see our paper [
SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans](https://arxiv.org/pdf/2006.14660).

[<img src="spsg.jpg">](https://arxiv.org/abs/2006.14660)


## Code
### Installation:  
Training is implemented with [PyTorch](https://pytorch.org/). This code was developed under PyTorch 1.2.0 and Python 2.7.

Please compile the extension modules by running the `install_utils.sh` script.


### Training:  
* See `python train.py --help` for all train options. 
* Example command: `python train.py --gpu 0 --data_path ./data/data-geo-color --frame_info_path ./data/data-frames --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --save_epoch 1 --save logs/mp --max_epoch 5`
* Trained model: [spsg.pth](http://kaldir.vc.in.tum.de/adai/SPSG/spsg.pth) (7.5M)

### Testing
* See `python test_scene_as_chunks.py --help` for all test options. 
* Example command: `python test_scene_as_chunks.py --gpu 0 --input_data_path ./data/mp_sdf_2cm_input --target_data_path ./data/mp_sdf_2cm_target --test_file_list ../filelists/mp-rooms_val-scenes.txt --model_path spsg.pth --output ./output --max_to_vis 20`

### Data:
* Scene data: 
  - [mp_sdf_2cm_input.zip](http://kaldir.vc.in.tum.de/adai/SPSG/mp_sdf_2cm_input.zip) (68G)
  - [mp_sdf_2cm_target.zip](http://kaldir.vc.in.tum.de/adai/SPSG/mp_sdf_2cm_target.zip) (87G)
* Train data:
  - [data-geo-color.zip](http://kaldir.vc.in.tum.de/adai/SPSG/data-geo-color.zip) (110G)
  - [data-frames.zip](http://kaldir.vc.in.tum.de/adai/SPSG/data-frames.zip) (11M)
  - [images.zip](http://kaldir.vc.in.tum.de/adai/SPSG/images.zip) (12G)
* Data generation code: [datagen](datagen)
  - Developed in Visual Studio 2017
  - Dependencies: Microsoft DirectX SDF June 2010, [mLib](https://github.com/niessner/mLib)
  - Downlaod the [Matterport3D](https://github.com/niessner/matterport) dataset and edit the corresponding file paths in [datagen/zParametersScanMP.txt](datagen/zParametersScanMP.txt). Note that the mesh files must have the face attributes removed to generate the `*.reduced.ply` files.

## Citation:  
If you find our work useful in your research, please consider citing:
```
@inproceedings{dai2021spsg,
    title={SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans},
    author={Dai, Angela and Siddiqui, Yawar and Thies, Justus and Valentin, Julien and Nie{\ss}ner, Matthias},
	booktitle={Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
	year={2021}
}
```
