import sys
sys.path.append('../../../')
import argparse
import scipy.io as sio

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

mat = sio.loadmat(args.input)['T']

print(mat.keys())

