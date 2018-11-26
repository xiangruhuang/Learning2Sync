import os
import numpy as np
import cv2
import sys
import argparse
import pathlib
import glob
import time
sys.path.append('../../')
from util import env, inverse, project_so, make_dirs
from mesh import Mesh
import scipy.io as sio

"""
    Draw a 3 by n point cloud using open3d library
"""
def draw(vertex):
    import open3d
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(vertex.T)
    open3d.draw_geometries([pcd])

def parse_info(path):
    with open(path, 'r') as fin:
        lines = fin.readlines()
        info_dict = {}
        for line in lines:
            key, val = line.strip().split('=')
            key = key.strip()
            val = val.strip()
            if key == 'm_calibrationDepthIntrinsic':
                val = [float(token) for token in val.split(' ')]
                val = np.reshape(val, (4, 4))
                assert (abs(val[:3, 3]) < 1e-4).all(), 'intrinsic :3, 3'
                assert (abs(val[3, :3]) < 1e-4).all(), 'intrinsic 3, :3'
                assert (abs(val[3, 3] - 1.0) < 1e-4).all(), 'intrinsic 3 3'
                val = val[:3, :3]
                return val

data_path = env()
print('home directory = %s' % data_path)

PATH_POSE = '%s/dataset/scannet/{}/{}.pose.txt' % data_path
PATH_DEPTH = '%s/dataset/scannet/{}/{}.depth.pgm' % data_path
PATH_MAT = '%s/processed_dataset/scannet/{}/{}.mat' % data_path

parser = argparse.ArgumentParser(description='Process Redwood Dataset')
parser.add_argument('--shapeid', type=str)

args = parser.parse_args()

def getData(shapeid):
    depth_paths = []
    poses = []
    pose_paths = []
    frames = glob.glob(PATH_DEPTH.format(shapeid, '*'))
    frames.sort()
    for i, frame in enumerate(frames):
        frameid = frame.split('/')[-1].split('.')[0]
        depth_path = PATH_DEPTH.format(shapeid, frameid)
        #tmp = cv2.resize(cv2.imread(imgsPath, 2)/1000., (64,64))
        #AuthenticdepthMap.append(tmp.reshape(1,tmp.shape[0],tmp.shape[1],1))

        pose_fp = PATH_POSE.format(shapeid, frameid)
        flag = True
        try:
            tmp = np.loadtxt(pose_fp)
            assert abs(tmp[3, 3] - 1.0) < 1e-4, 'bottom right corner should be one'
            assert (abs(tmp[3, :3]) < 1e-4).all(), '[3, :3] should be zero'
            R = tmp[:3, :3]
            assert np.linalg.det(R) > 0.01, 'determinant should be 1'
            assert np.linalg.norm(R.dot(R.T) - np.eye(3), 'fro') ** 2 < 1e-4, 'should be a rotation matrix'
            project_R = project_so(R)
            assert np.linalg.norm(R-project_R, 'fro') ** 2 < 1e-4, 'projection onto SO3 should be identical'
            tmp[:3, :3] = project_R
            tmp = inverse(tmp)
        except Exception as e:
            print('error on {}: {}'.format(pose_fp, e))
            #print(R.dot(R.T))
            #print(np.linalg.norm(R.dot(R.T) - np.eye(3), 'fro'))
            flag = False
        if not flag:
            print('ignoring frame {}'.format(frameid))
            assert False
        poses.append(tmp)
        depth_paths.append(depth_path)
        pose_paths.append(pose_fp)

    T = np.concatenate(poses).reshape(-1,4,4)
    return depth_paths, T, pose_paths

def main():
    depth_paths, T, pose_paths = getData(args.shapeid)
    n = len(depth_paths)
    print('found %d clean depth images...' % n)
    intrinsic = parse_info('%s/dataset/scannet/%s/_info.txt' % (data_path, args.shapeid))
    np.random.seed(816)
    num_sample = 100
    if n < 10*num_sample:
        stepsize = n // num_sample
    else:
        stepsize = 10
    if stepsize == 0:
        assert False
    indices = [i for i in range(0, n, stepsize)][:num_sample] #np.random.permutation(n)
    #print(indices[:100])
    #indices = sorted(indices)
    make_dirs(PATH_MAT.format(args.shapeid, 0))
    #import open3d
    #pcd_combined = open3d.PointCloud()
    for i, idx in enumerate(indices):
        print('%d / %d' % (i, len(indices)))
        mesh = Mesh.read(depth_paths[idx], mode='depth', intrinsic = intrinsic)
        #pcd = open3d.PointCloud()
        #pcd.points = open3d.Vector3dVector(mesh.vertex.T)
        #pcd.transform(inverse(T[idx]))
        ##pcd = open3d.voxel_down_sample(pcd, voxel_size=0.02)
        #pcd_combined += pcd
        #pcd_combined = open3d.voxel_down_sample(pcd_combined, voxel_size=0.02)
        sio.savemat(PATH_MAT.format(args.shapeid, i), mdict={
            'vertex': mesh.vertex, 
            'validIdx_rowmajor': mesh.validIdx, 
            'pose': T[idx], 
            'depth_path': depth_paths[idx], 
            'pose_path': pose_paths[idx]})
        #if (i + 1) % 100 == 0:
        #    pcd_combined_down = open3d.voxel_down_sample(pcd_combined, voxel_size=0.02)
        #    open3d.draw_geometries([pcd_combined_down])
            

    #pcd_combined_down = open3d.voxel_down_sample(pcd_combined, voxel_size=0.02)
    #open3d.draw_geometries([pcd_combined_down])
        #draw(mesh.vertex)
    #sId = np.kron(np.array(range(n)), np.ones([n,1])).astype('int')
    #tId = np.kron(np.array(range(n)).reshape(-1,1), np.ones([1,n])).astype('int')
    #valId = (sId > tId)
    #sId = sId[valId]
    #tId = tId[valId]
    #numEach = 1
    #print('n=%d' % n)
    #print('numEach=%d' % numEach)
    #left = numEach * args.split
    #right = min(numEach * (1 + args.split), len(sId))
    #print('computing [%d:%d] out of [%d:%d]' % (left, right, 0, len(sId)))
    #sId = sId[left:right]
    #tId = tId[left:right]
    #
    #for i in range(len(sId)):
    #    sId_this = sId[i]
    #    tId_this = tId[i]
    #    print(sId_this, tId_this)
    #    sys.stdout.flush()
    #    outpath = os.path.join(outDir, '{}_{}.npy'.format(sId_this,tId_this))
    #    #if os.path.exists(outpath):
    #    #    continue
    #    
    #    start_time = time.time()
    #    """
    #    sourceMeshNPY = convertMatlabFormat(DepthPath[sId_this])[np.newaxis,:]
    #    targetMeshNPY = convertMatlabFormat(DepthPath[tId_this])[np.newaxis,:]
    #    #import pdb; pdb.set_trace()
    #    print('convert')
    #    sys.stdout.flush()
    #    validId = (sourceMeshNPY.sum(2)!=0).squeeze()
    #    import util
    #    util.pc2obj(sourceMeshNPY[0,validId,:].T,'test1.obj')
    #    util.pc2obj(targetMeshNPY[0,validId,:].T,'test2.obj')
    #    print('source, target')
    #    print('time elapsed = %f' % (time.time() - start_time))
    #    sys.stdout.flush()

    #    sourceMesh = matlab.double(sourceMeshNPY.tolist())
    #    targetMesh = matlab.double(targetMeshNPY.tolist())
    #    #import pdb; pdb.set_trace()
    #    print('time elapsed = %f' % (time.time() - start_time))
    #    R_,t_,sigma=eng.pythonMain(sourceMesh,targetMesh,nargout=3)
    #    R = np.zeros([4,4])
    #    R[:3,:3] = np.array(R_)
    #    R[3,3] = 1
    #    R[:3,3] = np.array(t_).squeeze()
    #    
    #    #sourceMeshNPYHomo = np.ones([4,sourceMeshNPY.shape[1]])
    #    #sourceMeshNPYHomo[:3,:] = sourceMeshNPY[0].copy().T
    #    #sourceMeshNPYHomo = np.matmul(R, sourceMeshNPYHomo)[:3,:]
    #    #util.pc2obj(sourceMeshNPYHomo,'test1T.obj')
    #    """
    #    sourceMesh = Mesh.read(DepthPath[sId_this],mode='depth',intrinsic=intrinsic)
    #    print(sourceMesh.vertex.shape)
    #    print('done loading source')
    #    sys.stdout.flush()
    #    targetMesh = Mesh.read(DepthPath[tId_this],mode='depth',intrinsic=intrinsic)
    #    print('done loading target')
    #    sys.stdout.flush()
    #    #np.save('temp.npy', {'R':Pose[tgt][:3, :3].dot, 'src': sourceMesh.vertex, 'tgt': targetMesh.vertex, 'srcValidIdx': sourceMesh.validIdx, tgtValidIdx: targetMesh.validIdx})
    #    #assert False
    #    R,sigma = globalRegistration(sourceMesh, targetMesh, optsRGBD())
    #    print('done registration')
    #    ##import ipdb
    #    ##ipdb.set_trace()
    #    
    #    print('dumping to %s' % outpath)
    #    np.save(outpath, {'R':R, 'sigma':sigma})
    #    end_time = time.time()
    #    print('time elapsed = %f' % (end_time - start_time))
    #    sys.stdout.flush()

    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)

if __name__ == '__main__':
    main()
