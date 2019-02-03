#!/bin/bash
export PATH=${PATH}:/usr/local/bin
source /u/xrhuang/anaconda3/etc/profile.d/conda.sh
export PATH=/u/xrhuang/anaconda3/bin:${PATH}
export PKG_CONFIG_PATH=/u/xrhuang/anaconda3/envs/py36/lib/pkgconfig/
source activate py36

python compute_sigmas.py --pid $1
