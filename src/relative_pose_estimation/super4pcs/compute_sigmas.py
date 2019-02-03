import argparse
import glob
import os
import pathlib
import sys
import numpy as np
sys.path.append('../../../')
from util import env, read_super4pcs, angular_distance_np, inverse
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN

def parse_args():
  parser = argparse.ArgumentParser(
    description='Compute median error within overlapping region')
  parser.add_argument('--dataset', 
    type=str,
    help='redwood or scannet',
    default='scannet')
  parser.add_argument('--pid', 
    type=int,
    default=None)
  parser.add_argument('--source',
    type=str,
    default='super4pcs')

  args = parser.parse_args()

  return args

def compute_sigma(mat_file1, mat_file2, txt, output_mat):
  mat1 = sio.loadmat(mat_file1)
  mat2 = sio.loadmat(mat_file2)
  v1 = mat1['vertex'] # [3, n]
  v2 = mat2['vertex'] # [3, n]
  Tij = read_super4pcs(txt)
  Tij_gt = mat2['pose'].dot(inverse(mat1['pose']))
  Rij = Tij[:3, :3]
  tij = Tij[:3, 3]
  v1 = Rij.dot(v1) + tij[:, np.newaxis]
  
  tree = NN(n_neighbors=1, algorithm='kd_tree').fit(v1.T)
  distances, _ = tree.kneighbors(v2.T)
  distances = distances[distances < 0.2]
  d = {}
  d['sigma'] = np.median(distances)
  d['Tij'] = Tij
  d['aerr'] = angular_distance_np(Tij[np.newaxis, :3, :3], Tij_gt[np.newaxis, :3, :3]).sum()
  d['terr'] = np.linalg.norm(Tij[:3, 3]- Tij_gt[:3, 3], 2)
  d['src'] = mat_file1
  d['tgt'] = mat_file2
  sio.savemat(output_mat, mdict=d, do_compression=True)

def main():
  args = parse_args()
  pid = args.pid
  source = args.source
  dataset = args.dataset

  data_path = env()
  
  PATH_MODEL='%s/processed_dataset/{}/{}/' % data_path
  PATH_RELATIVE = '%s/relative_pose/{}' % data_path
  
  models = [os.path.normpath(p) for p in glob.glob(PATH_MODEL.format(dataset, '*'))]
  models.sort()
  for model in models[args.pid::100]:
    print(model)
    objs = glob.glob('%s/*.mat' % model)
    modelname = ('/').join(model.split('/')[-2:])
    
    n = len(objs)
    basename = [int(obji.split('/')[-1].split('.')[0]) for obji in objs]

    output_folder = PATH_RELATIVE.format(modelname)
    
    for i in range(n):
      for j in range(i+1, n):
        txt_file = '{}/{}_{}_super4pcs.txt'.format(output_folder, basename[i], basename[j])
        mat_file = '{}/{}_{}_super4pcs.mat'.format(output_folder, basename[i], basename[j])
        compute_sigma(objs[i], objs[j], txt_file, mat_file)
        

if __name__ == '__main__':
  main()
