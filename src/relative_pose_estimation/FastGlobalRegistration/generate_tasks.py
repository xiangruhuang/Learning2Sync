import glob
import os
import pathlib
import sys
import numpy as np
sys.path.append('../../../')
from util import env, make_dirs

dataset = 'scannet'

def main():
    data_path = env()

    PATH_MODEL='%s/processed_dataset/{}/{}/' % data_path
    PATH_RELATIVE = '%s/relative_pose/{}' % data_path

    models = [os.path.normpath(p) for p in glob.glob(PATH_MODEL.format(dataset, '*'))]
    make_dirs('%s/tasks' % dataset)
    with open('%s/tasks' % dataset, 'w') as fout:
        lines = []
        for model in models:
            objs = glob.glob('%s/*' % model)
            modelname = ('/').join(model.split('/')[-2:])
            print(modelname)
            #print(modelname)
            #import ipdb; ipdb.set_trace()
            n = len(objs)
            basename = [obji.split('/')[-1].split('.')[0] for obji in objs]
            
            for i in range(n):
                for j in range(i+1, n):
                    #if np.random.rand() > 0.01:
                    #    continue
                    output_folder = PATH_RELATIVE.format(modelname) 
                    #'%s/relative_pose/%s' % (data_path, modelname)
                    pathlib.Path(output_folder).mkdir(exist_ok = True, parents = True)
                    output_file = '%s/%s_%s_fgr.mat' % (output_folder, basename[i], basename[j])
                    
                    command = 'python global_registration.py %s %s %s' % (objs[i], objs[j], output_file)
                    #print(command)
                    lines.append(command)
                    #fout.write('%s\n' % command)
                    """ ./Super4PCS -i ../datasets/redwood/00021_0.obj ../datasets/redwood/00021_1.obj -o 0.7 -d 0.01 -t 1000 -n 200 -m super4pcs/00021_0_1.txt """
            break
        #np.random.shuffle(lines)
        for line in lines:
            fout.write('%s\n' % line)

if __name__ == '__main__':
    main()
