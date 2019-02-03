import glob
import os
import pathlib
import sys
import numpy as np
sys.path.append('../../../')
from util import env, make_dirs
import subprocess

dataset = 'scannet'

param ='-o 0.7 -d 0.01 -t 1000 -n 200'

def main():
    data_path = env()

    PATH_MODEL='%s/processed_dataset/{}/{}/' % data_path
    PATH_RELATIVE = '%s/relative_pose/{}' % data_path

    models = [os.path.normpath(p) for p in glob.glob(PATH_MODEL.format(dataset, '*'))]
    for model in models:
        if not 'scene0224_00' in model:
          if not 'scene0622_00' in model:
            continue
        objs = glob.glob('%s/*.obj' % model)
        modelname = ('/').join(model.split('/')[-2:])
        n = len(objs)
        basename = [int(obji.split('/')[-1].split('.')[0]) for obji in objs]
        
        output_folder = PATH_RELATIVE.format(modelname)
        for i in range(n):
            for j in range(i+1, n): 
                output_file = '{}/{}_{}_super4pcs.txt'.format(output_folder, basename[i], basename[j])
                if not os.path.exists(output_file): 
                  command = './Super4PCS -i %s %s %s -m %s' % (objs[i], objs[j], param, output_file)
                  print(command)
                  os.system(command)
                """ ./Super4PCS -i ../datasets/redwood/00021_0.obj ../datasets/redwood/00021_1.obj -o 0.7 -d 0.01 -t 1000 -n 200 -m super4pcs/00021_0_1.txt """

if __name__ == '__main__':
    main()
