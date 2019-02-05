import numpy as np
import glob
import sys
import scipy.io as sio
import argparse
sys.path.append('../../')
from util import env, decompose, angular_distance_np, inverse, Reader

parser = argparse.ArgumentParser(
  description='measure error of input')
parser.add_argument('--dataset',
  type=str, help='redwood or scannet',
  default='redwood')
parser.add_argument('--source',
  type=str, help='fgr or super4pcs',
  default='fgr')

args = parser.parse_args()

data_path = env()

dataset = args.dataset
source = args.source

with open('%s/experiments/%s.test' % (data_path, dataset), 'r') as fin:
  lines = [line.strip() for line in fin.readlines()]
  print(lines)

with open('%s/experiments/%s.train' % (data_path, dataset), 'r') as fin:
  lines2 = [line.strip() for line in fin.readlines()]
  lines = lines + lines2

terrs = []
aerrs = []
sigmas = []

for line in lines:
  summary_mat = '%s/relative_pose/summary/%s/%s/%s.mat' % (data_path, dataset, source, line)
  summary_mat = sio.loadmat(summary_mat)
  T = summary_mat['T']
  Tstar = summary_mat['Tstar']
  aerr = summary_mat['aerr']
  terr = summary_mat['terr']
  sigma = summary_mat['sigma']
  n = Tstar.shape[0]
  n = 30
  for i in range(n):
    for j in range(i+1, n):
      Tij = T[i*4:(i+1)*4, j*4:(j+1)*4]
      Tij_gt = Tstar[j, :, :].dot(inverse(Tstar[i, :, :]))
      terr_ij = np.linalg.norm((Tij_gt - Tij)[:3, 3], 2)
      assert abs(terr_ij - terr[i, j]) < 1e-4
      terrs.append(terr_ij)
      aerr_ij = angular_distance_np(Tij_gt[np.newaxis, :3, :3], Tij[np.newaxis, :3, :3]).sum()
      assert abs(aerr_ij - aerr[i, j]) < 1e-4
      aerrs.append(aerr_ij)
      sigmas.append(sigma[i, j])

aerrs = np.array(aerrs)
terrs = np.array(terrs)
sigmas = np.array(sigmas)

for sigma_threshold in [0.1, 0.2]:
  valid_indices = np.where(sigmas < sigma_threshold)[0]
  
  terrs_temp = terrs[valid_indices]
  aerrs_temp = aerrs[valid_indices]
  
  for a in [3.0, 5.0, 10.0, 30.0, 45.0]:
    p = len(np.where(aerrs_temp < a)[0]) * 1.0 / len(aerrs_temp)
    print('Rotation: \tpercentage below %f = %f' % (a, p))
  print('Rotation: Mean=%f' % np.mean(aerrs_temp))

  for t in [0.05, 0.1, 0.25, 0.5, 0.75]:
    p = len(np.where(terrs_temp < t)[0]) * 1.0 / len(terrs_temp)
    print('Translation: \tpercentage below %f = %f' % (t, p))
  print('Translation: \tMean=%f' % np.mean(terrs_temp))

