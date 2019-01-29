import numpy as np
import scipy.io as sio
import sys
sys.path.append('../../')
from util import env
import glob
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Visualizing Graph Structure')
parser.add_argument('--dataset', type=str)
parser.add_argument('--shapeid', type=str)

args = parser.parse_args()

home = env()

PATH_FGR = '%s/relative_pose/{}/{}/{}.mat' % home
PATH_POSE = '%s/processed_dataset/{}/{}/{}.mat' % home

mats = glob.glob(PATH_FGR.format(args.dataset, args.shapeid, '*'))
n = 100
aerr = np.zeros((n, n))
terr = np.zeros((n, n))
top = 0.0
bottom = 0.0
for mat_file in mats:
    mat = sio.loadmat(mat_file)
    #print(mat.keys())
    x, y = [int(token) for token in mat_file.split('/')[-1].split('.')[0].split('_')[:2]]
    if x > y:
        tmp = x; x = y; y = tmp
    mat_x = sio.loadmat(PATH_POSE.format(args.dataset, args.shapeid, x))['depth_path']
    #print(x, str(mat_x).split('\'')[1].split('/')[-1].split('.')[0])
    #print(mat['terr'], mat['aerr'])
    if mat['terr'] < 0.2 and mat['aerr'] < 15.0:
        aerr[x, y] = 1.0
        top += 1.0
    else:
        aerr[x, y] = 0.0
    bottom += 1.0
    #terr[x, y] = mat['terr']
    #aerr[x, y] = mat['aerr']
   
#X = np.array(range(n))
#Y = np.array(range(n))
#X, Y = np.meshgrid(X, Y)

print(top / bottom)

plt.imshow(aerr)
plt.colorbar()
plt.show()
