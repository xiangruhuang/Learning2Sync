import sys
sys.path.append('../../../')
from util import env
import argparse
import scipy.io as sio
from TransfSync import IterativeTransfSync, errors
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='Perform Transformation Synchronization')
    parser.add_argument('--input_list', 
        type=str, default=None, help='list of input .mat file')
    parser.add_argument('--truncate',
        action='store_true', default=None,
        help='using truncated scheme {False}')
    parser.add_argument('--reweight',
        action='store_true', default=None,
        help='using reweight scheme {True}')
    parser.add_argument('--cheat', 
        action='store_true', default=False, 
        help='use ground truth labels (for debug)')
    parser.add_argument('--label',
        type=str, default=None,
        help='path to .mat file generated, {classification/results/14.p}')
    parser.add_argument('--output',
        type=str, default='temp.mat',
        help='output path, must be a .mat file {temp.mat}')
    args = parser.parse_args()

    if args.truncate is None:
        args.truncated = False

    if args.reweight is None:
        args.reweighted = False

    if args.truncate and args.reweight:
        raise ValueError('both truncate and reweight schemes are specified')

    if (not args.truncate) and (not args.reweight):
        raise ValueError('need to specify exactly one between "truncate" and "reweight" ')
    
    if args.input_list is None:
        home = env()
        args.input_list = 'input_list'

    if args.reweight:
        args.scheme = 'reweight'
    if args.truncate:
        args.scheme = 'truncate'

    if args.input_list is None:
        raise ValueError('input file needs to be specified')
    return args

def from_mat(mat_file, label_dict=None, cheat=False, scheme='reweight'):
    mat = sio.loadmat(mat_file)

    T = mat['T']
    Tstar = mat['Tstar']
    n = Tstar.shape[0]
    edges = []
    
    #if args.label is not None:
    #    with open(args.label, 'rb') as fin:
    #        predictions = pickle.load(fin)
    #    #prediction = pickle.load(args.label)
    #    gts = prediction['gt'][0]
    #    labels = prediction['predict'][0]
    #    files = [('/').join(f.strip().split('/')[-2:]).split('.')[0] for f in prediction['files']]
    #    files = [('_').join(f.split('_')[:-2]) for f in files] 
    #    label_dict = {f.split('/')[1]: label for label, f in zip(labels, files) if (f.split('/')[0] in mat_file)}
    #    gt_dict = {f.split('/')[1]: gt for gt, f in zip(gts, files) if (f.split('/')[0] in mat_file)}
    #    diffs = []
    #    for ff in files:
    #        f = ff.split('/')[1] 
    #        a = np.round(label_dict.get(f, 0.0))
    #        b = np.round(gt_dict.get(f, 0.0))
    #        diff = abs(a - b)
    #        diffs.append(diff)
    #    print('classifier error = %f' % np.mean(diffs))
                
    for i in range(n):
        for j in range(i+1, n):
            edge = {}
            edge['src'] = i
            edge['tgt'] = j
            Tij = T[i*4:(i+1)*4, j*4:(j+1)*4]
            if abs(Tij[3, 3] - 1.0) > 1e-3:
                continue
            R = Tij[:3, :3]
            assert np.linalg.norm(R.dot(R.T) - np.eye(3), 'fro') < 1e-3
            assert np.linalg.det(R) > 0.01
            edge['R'] = R
            edge['t'] = Tij[:3, 3]
            if label_dict is not None:
                weight = np.clip(label_dict[i, j], 0.0, 1.0)
            else:
                weight = 1.0
            edge['rotation_weight'] = weight
            edge['translation_weight'] = weight
            edge['predicted_weight'] = weight
            edges.append(edge)
    print('#edges=%d' % len(edges))
    Tsync = IterativeTransfSync(n, edges, Tstar = Tstar, cheat=cheat, scheme=scheme, max_iter=5)
    aerrs, terrs = errors(Tsync, Tstar)
    return aerrs, terrs

def main():
    args = parse_args()
    aerrs = []
    terrs = []
    with open(args.input_list, 'r') as fin:
        mats = [line.strip() for line in fin.readlines()]
    if args.label is not None:
        with open(args.label, 'rb') as fin:
            predictions = pickle.load(fin)
    else:
        predictions = {}

    home = env()
    PATH_MAT = '%s' % home
    diffs = []
    for mat_file in mats:
        print(mat_file)
        scene = mat_file.split('/')[-1].split('.')[0]
        label_dict = predictions.get(scene, None)
        if label_dict is not None:
            indices = np.triu_indices(100, 1)
            a = np.round(label_dict['predict'][indices])
            b = np.round(label_dict['gt'][indices])
            print(a.shape)
            diff = abs(a - b)
            diffs.append(diff)
            print('%s: classifier error=%f' % (scene, np.mean(diff)))
            label_dict = label_dict['predict']
            
        aerr, terr = from_mat(mat_file, label_dict, args.cheat, args.scheme)
        aerrs.append(aerr)
        terrs.append(terr)
    if len(diffs) > 0:
        diffs = np.concatenate(diffs, axis=0)
        print('classifier error=%f' % (np.mean(diffs)))

    aerrs = np.concatenate(aerrs, axis=0)
    terrs = np.concatenate(terrs, axis=0)
    name = args.output
    print('dumping to %s' % name)
    sio.savemat('%s' % name, mdict={'aerrs': aerrs, 'terrs': terrs})

if __name__ == '__main__':
    main()
