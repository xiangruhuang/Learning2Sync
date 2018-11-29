import sys
import glob
import os
from sklearn.neighbors import NearestNeighbors as NN
import pathlib
sys.path.append('../../')
from util import inverse, env, angular_distance_np, decompose
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import numpy as np

home = env()

def __get_label__(Rij, tij, Ti, Tj):
    """ Measure Quality of Edge """
    Ristar, tistar = decompose(Ti)
    Rjstar, tjstar = decompose(Tj)

    label = 0.0
    err_R = angular_distance_np(Rij[np.newaxis, :, :], Rjstar.dot(Ristar.T)[np.newaxis, :, :]).sum()
    err_T = np.linalg.norm(Rij.dot(tistar) + tij - tjstar, 2)
    if err_R < 30.0:
        label = 1.0
    else:
        label = 0.0
    return label

def generate_pair(Tij, sid, tid, vi, vj, Ti, Tj, idx1, idx2, gt, suffix):
    width = 640; height = 480
    #assert width == mtgt.width
    #assert height == mtgt.height
    assert height == 480
    assert width == 640
    
    idx1 = idx1[0, :]
    idx2 = idx2[0, :]
    tree_tgt = NN(n_neighbors=1, algorithm='kd_tree').fit(vj.T)
    #Tij = icp(vi.T, vj.T, init_pose=Tij, dmax=0.2, tree_tgt = tree_tgt, numSamples=2000)
    Rij, tij = decompose(Tij)
    label = __get_label__(Rij, tij, Ti, Tj)
    vi = (Rij.dot(vi).T + tij).T
    #print(src, tgt, label)
    #""" Compute Dist Image for Image 1 """
    #dist1, index1 = tree[tgt].kneighbors(v1)
    #dist2, index2 = tree[src].kneighbors(v2)
    #print([np.mean(sorted(dist1)[:10000*i]) for i in range(1, 8)])
    #print([np.mean(sorted(dist2)[:10000*i]) for i in range(1, 8)])
    tree_src = NN(n_neighbors=1, algorithm='kd_tree').fit(vi.T)
    
    #v2 = Rij.T.dot().T - Rij.T.dot(tij); idx2 = mtgt.validIdx
    dist1, index1 = tree_tgt.kneighbors(vi.T)
    dist2, index2 = tree_src.kneighbors(vj.T)
    
    #import ipdb; ipdb.set_trace()
    image1 = np.zeros(width * height) + dist1.max()
    #print(dist1.max())
    image1[idx1] = dist1[:, 0]
    #image1 = (image1 - image1.min()) / (image1.max() - image1.min()) * 255.0
    image1 = np.power(image1, 0.25)
    image1 = np.reshape(image1, [height, width])
    #print('frac of hole = %f' % fracOfHole)
    
    """ Compute Dist Image for Image 2 """
    #print(dist2.max())
    image2 = np.zeros(width * height) + dist2.max()
    image2[idx2] = dist2[:, 0]
    #image2 = (image2 - image2.min()) / (image2.max() - image2.min()) * 255.0
    image2 = np.power(image2, 0.25)
    image2 = np.reshape(image2, [height, width])
    #print('frac of hole = %f' % fracOfHole)
    
    #######################################################################
    #""" Save Figure """
    #image_concat = np.concatenate((image1, image2), axis=1)
    #
    ##pathlib.Path('knn_images_%d_%d_%s' % (sid, tid, gt)).mkdir(parents=True, exist_ok=True)
    #plt.imshow(image_concat, cmap='hot')
    ##plt.colorbar()
    #if gt: 
    #    plt.savefig('knn_images/%d_%d_%d_%s_gt.png' % (label, sid, tid, suffix))
    #else:
    #    plt.savefig('knn_images/%d_%d_%d_%s_recover.png' % (label, sid, tid, suffix))
    #######################################################################
    
    image = np.stack((image1, image2), axis=0)
    assert image.shape == (2, height, width)
    return image, label, Tij

def get_scanids(model):
    model_path = PATH_SEQ.format(model)
    scans = glob.glob('%s/*.mat' % model_path)
    scanids = [int(scan.split('/')[-1].split('.')[0]) for scan in scans]
    scanids = sorted(scanids)
    return scanids 

def read_scan(model, matid):
    scanids = get_scanids(model)
    model_path = PATH_SEQ.format(model)
    mat = '%s/%d.mat' % (model_path, scanids[matid])
    mat = sio.loadmat(mat)
    return mat
    
def read_transformation(model, suffix):
    assert 2 == len(model.split('/'))
    dataset = model.split('/')[0]
    shapeid = model.split('/')[-1]
    scanids = get_scanids(model)
    rel = glob.glob('%s/relative_pose/summary_%s/%s/%s.mat' % (data_path, suffix, dataset, shapeid))
    assert len(rel) == 1
    mat = sio.loadmat(rel[0])
    return mat

def main():
    parser = argparse.ArgumentParser(description='Generate Images for Network Training')
    parser.add_argument('--dataset', type=str, default='scannet')
    parser.add_argument('--modelid', type=int)
    parser.add_argument('--source', type=str, default='fgr')
    args = parser.parse_args()
    with open('%s.list' % args.dataset, 'r') as fin:
        shapelist = [line.strip() for line in fin.readlines()]
        args.shapeid = shapelist[args.modelid]
    source =  args.source
    PATH_RELATIVE = '%s/relative_pose/%s/%s/{}.mat' % (home, args.dataset, args.shapeid)
    PATH_ABS = '%s/processed_dataset/%s/%s/{}.mat' % (home, args.dataset, args.shapeid)
    mats = glob.glob(PATH_RELATIVE.format('*'))
    np.random.shuffle(mats)
    model = '%s/%s' % (args.dataset, args.shapeid)
    n = 100
    v = []
    idx = []
    T = []
    for i in range(n):
        mat_i = sio.loadmat(PATH_ABS.format(i))
        v.append(mat_i['vertex'])
        idx.append(mat_i['validIdx_rowmajor'])
        T.append(mat_i['pose'])

    """ Creating Data Folders """
    dump_folder = '%s/classifier_training/%s/%s/' % (home, args.dataset, args.shapeid)
    if args.dataset == 'redwood':
        target_folder = '/media/xrhuang/DATA/%s/%s' % (args.dataset, args.shapeid)
    else:
        target_folder = '/media/xrhuang/DATA1/%s/%s' % (args.dataset, args.shapeid)
    pathlib.Path(dump_folder).mkdir(exist_ok=True, parents=True)
    command = 'ssh xrhuang@qhgroup-desktopi.csres.utexas.edu mkdir -p %s' % target_folder
    res = os.system(command)
    while res != 0:
        res = os.system(command)
    
    for mat_file in mats:
        i, j = [int(token) for token in mat_file.split('/')[-1].split('.')[0].split('_')[:2]]
        mat = sio.loadmat(mat_file)
        Tij = mat['Tij']
        if i > j:
            tmp = i; i = j; j = tmp
            Tij = inverse(Tij)
        
        assert abs(Tij[3, 3] - 1.0) < 1e-6

        Ti = T[i]; vi = v[i]; idx1 = idx[i]
        Tj = T[j]; vj = v[j]; idx2 = idx[j]
        
        assert vi.shape[0] == 3
        name = '%d_%d_%s_recover.mat' % (i, j, source)
        
        succeed = True
        image, label, Tij_icp = generate_pair(Tij, i, j, vi, vj, Ti, Tj, idx1, idx2, False, source)
        sio.savemat('%s/%s' % (dump_folder, name), {'image': image, 'label': label, 'Tij_icp': Tij_icp}, do_compression=False)
        command = 'scp %s xrhuang@qhgroup-desktopi.csres.utexas.edu:%s/' % ('%s/%s' % (dump_folder, name), target_folder)
        print(command)
        res = os.system(command)
        while res != 0:
            print('Failed at scp, command=%s' % command)
            res = os.system(command)
        command = 'rm %s/%s' % (dump_folder, name)
        os.system(command)
        

 
if __name__ == '__main__':
    main()
