import os, sys
import pathlib
import glob
import scipy.io as sio
sys.path.append('../../')
from util import env, inverse, Reader
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Summarize relative pose estimations into .mat format')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--source', type=str, default=None)
args = parser.parse_args()

if args.dataset is None:
    raise ValueError('Must specify dataset, e.g. redwood or scannet, etc.')
if args.source is None:
    raise ValueError('Must specify input source, e.g. fgr or Super4PCS, etc.')

home = env()
dataset = args.dataset
source = args.source

pathlib.Path('%s/relative_pose/summary/%s/%s' % (home, dataset, source)).mkdir(
    exist_ok=True, parents=True)
reader = Reader()
PATH_SUMMARY = '%s/relative_pose/summary/{}/{}/{}.mat' % home
for sceneid in reader.list_scenes(dataset):
    scanids = reader.get_scanids(dataset, sceneid)
    
    n = len(scanids)
    scanid_map = {str(scanid): i for i, scanid in enumerate(scanids)}
    T = np.zeros((n*4, n*4))
    sigma = np.zeros((n, n))
    aerr = np.zeros((n, n))
    terr = np.zeros((n, n))
    for mat in glob.glob('%s/%s/*_%s.mat' % (dataset, sceneid, source)):
        s = sio.loadmat(mat)
        src, tgt = mat.split('/')[-1].split('.')[0].split('_')[:2]
        sid = scanid_map[src]
        tid = scanid_map[tgt]
        Tij = s['Tij']
        if sid > tid:
            tmp = sid; sid = tid; tid = tmp
            Tij = inverse(Tij)
        assert sid < tid
        sigma[sid, tid] = s['sigma']
        aerr[sid, tid] = s['aerr']
        terr[sid, tid] = s['terr']
        T[sid*4:(sid+1)*4, tid*4:(tid+1)*4] = Tij
    
    Tstar = np.zeros((n, 4, 4))
    
    for i, scanid in enumerate(scanids):
        scan = reader.read_scan(dataset, sceneid, scanid, variable_names=['pose'])
        Tstar[i, :, :] = scan['pose']
    print(sceneid)
    output_mat = PATH_SUMMARY.format(dataset, source, sceneid)
    sio.savemat(output_mat, 
        mdict={'T': T, 'sigma': sigma, 
            'aerr': aerr, 'terr': terr, 
            'Tstar': Tstar}, 
        do_compression=True)
