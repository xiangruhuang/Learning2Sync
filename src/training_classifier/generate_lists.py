import glob
import sys
sys.path.append('../../')
from util import env
import numpy as np

path = '/media/xrhuang/DATA1/scannet'

models = glob.glob('%s/*' % path)
print(len(models))
np.random.shuffle(models)
split_point = (len(models) * 2) // 3
train_models = models[:split_point]
test_models = models[split_point:]

with open('%s/classification/train_list' % env(), 'w') as fout:
    for model in train_models:
        sceneid = model.split('/')[-1]
        mats = glob.glob('%s/%s/*.mat' % (path, sceneid))
        for mat in mats:
            fout.write('%s\n' % mat)

with open('%s/classification/test_list' % env(), 'w') as fout:
    for model in test_models:
        sceneid = model.split('/')[-1]
        mats = glob.glob('%s/%s/*.mat' % (path, sceneid))
        for mat in mats:
            fout.write('%s\n' % mat)
