import os, sys
import pathlib
import glob
import scipy.io as sio
sys.path.append('../')
from util import env, list_scenes, inverse
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Summarize relative pose estimations into compact format (.mat)')
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

pathlib.Path('summary_%s/%s/' % (source, dataset)).mkdir(exist_ok=True, parents=True)
for sceneid in list_scenes(dataset):
    scene = '%s/%s' % (dataset, sceneid)
    scans = glob.glob('%s/processed_dataset/%s/%s/*.mat' % (home, dataset, sceneid))
    scans.sort()
    scanids = [int(scan.split('/')[-1].split('.')[0]) for i, scan in enumerate(scans)]
    scanids = sorted(scanids)
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
    #import ipdb; ipdb.set_trace() 
    print(scene)
    sio.savemat('summary_%s/%s.mat' % (source, scene), mdict={'T': T, 'sigma': sigma, 'aerr': aerr, 'terr': terr}, do_compression=True)
