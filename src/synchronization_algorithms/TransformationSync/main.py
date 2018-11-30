import sys
sys.path.append('../../../')
import argparse
import scipy.io as sio
from TransfSync import IterativeTransfSync
import numpy as np

parser = argparse.ArgumentParser(
    description='Perform Transformation Synchronization')
parser.add_argument('--input', type=str, 
    default=None, help='input .mat file')
parser.add_argument('--truncate',
    action='store_true', default=None,
    help='using truncated scheme {False}')
parser.add_argument('--reweight',
    action='store_true', default=None,
    help='using reweight scheme {True}')
args = parser.parse_args()

if args.truncate is None:
    args.truncated = False

if args.reweight is None:
    args.reweighted = False

if args.truncate and args.reweight:
    raise ValueError('both truncate and reweight schemes are specified')

if (not args.truncate) and (not args.reweight):
    raise ValueError('need to specify exactly one between "truncate" and "reweight" ')

if args.reweight:
    args.scheme = 'reweight'
if args.truncate:
    args.scheme = 'truncate'

if args.input is None:
    raise ValueError('input file needs to be specified')

mat = sio.loadmat(args.input)

T = mat['T']
Tstar = mat['Tstar']
n = Tstar.shape[0]
edges = []
for i in range(n):
    for j in range(i+1, n):
        edge = {}
        edge['src'] = i
        edge['tgt'] = j
        Tij = T[i*4:(i+1)*4, j*4:(j+1)*4]
        assert abs(Tij[3, 3] - 1.0) < 1e-3
        R = Tij[:3, :3]
        assert np.linalg.norm(R.dot(R.T) - np.eye(3), 'fro') < 1e-3
        assert np.linalg.det(R) > 0.01
        edge['R'] = R
        edge['t'] = Tij[:3, 3]
        edge['rotation_weight'] = 1.0
        edge['translation_weight'] = 1.0
        edge['predicted_weight'] = 1.0
        edges.append(edge)

Tsync = IterativeTransfSync(n, edges, Tstar = Tstar, cheat=True, scheme=args.scheme)
