#!/bin/bash
export PATH=${PATH}:/usr/local/bin
source /u/xrhuang/anaconda3/etc/profile.d/conda.sh
export PATH=/u/xrhuang/anaconda3/bin:${PATH}
export PKG_CONFIG_PATH=/u/xrhuang/anaconda3/envs/py36/lib/pkgconfig/
source activate py36

offset=0
a=$(( $offset + $1 ))
python generate_dataset.py --modelid $a
