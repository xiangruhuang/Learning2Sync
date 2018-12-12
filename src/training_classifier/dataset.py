from torch.utils.data.dataset import Dataset
#from openRGBDCondor import getData 
import sys
import os
#from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
#sys.path.append('./classic_py/')
#from openRGBD.base import Mesh
#sys.path.append('../TransformationSync/')
#from TS import read_npys
import scipy.misc
import pathlib
#from openRGBD.registration import icp
#from icp import icp
#import matplotlib.pyplot as plt
import glob

import scipy.io as sio

def inverse(T):
    R, t = __decompose__(T)
    invT = np.zeros((4, 4))
    invT[:3, :3] = R.T
    invT[:3, 3] = -R.T.dot(t)
    invT[3, 3] = 1
    return invT

def __pack__(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T

def __decompose__(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

"""
    This class maintains a list of paths to samples, 
        each contains a (image, label) pair
"""
class MyDataset(Dataset):
    """
        Parse list file, each line contains a path to one sample file
    """
    def __init__(self, path2list):
        with open(path2list, 'r') as fin:
            lines = [line.strip() for line in fin.readlines()]
        self.files = lines

    """
        FEEL FREE TO MODIFY THIS :)

        Args:
        `index`: which sample file to read
            each sample file is assumed to be a .mat file,
            which reads into a python dictionary that contains keyword
            'image' and 'label'
        
        Returns:
        `image`: np.ndarray of shape (2, width, height)
        `label`: int
    """
    def __getitem__(self, index):
        flag = False
        while not flag:
            flag = True
            try:
                data_dict = sio.loadmat(self.files[index])
                image = data_dict['image']
                label = int(data_dict['label'])
            except Exception as e:
                print(e)
                print('%s is corrupted.' % self.files[index])
                flag = False
                index = index + 1
                if index >= len(self.files):
                    index = 0
            
        return (image, label, self.files[index])
        #return (self.images[index], self.labels[index])

    """
        Return the number of sample files maintained
    """
    def __len__(self):
        return len(self.files)
