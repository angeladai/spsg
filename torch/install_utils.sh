#!/bin/bash

cd utils
cd color_utils_cpu
python setup.py install
cd ..
cd depth_utils
python setup.py install
cd ..
cd marching_cubes
python setup.py install
cd ..
cd raycast_rgbd
python setup.py install
cd ../..
