import numpy as np
import copy
import scipy.io as sio
import sys
sys.path.append('../../../')
from util import env, inverse
from sklearn.neighbors import NearestNeighbors as NN
import open3d

def execute_fast_global_registration(source_down, target_down,
        source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = open3d.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            open3d.FastGlobalRegistrationOption(
            maximum_correspondence_distance = distance_threshold))
    return result

def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    n = R.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    open3d.estimate_normals(pcd_down, open3d.KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.compute_fpfh_feature(pcd_down,
            open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

#def prepare_dataset(voxel_size):
#    print(":: Load two point clouds and disturb initial pose.")
#    source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
#    target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
#    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
#                            [1.0, 0.0, 0.0, 0.0],
#                            [0.0, 1.0, 0.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]])
#    source.transform(trans_init)
#    draw_registration_result(source, target, np.identity(4))
#
#    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
#    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
#    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = open3d.registration_icp(source, target, distance_threshold,
            result_fast.transformation,
            open3d.TransformationEstimationPointToPlane())
    return result


if __name__ == "__main__":
    voxel_size = 0.02 # means 5cm for the dataset
    import argparse
    parser = argparse.ArgumentParser(description='Baseline Algorithm: Fast Global Registration')
    parser.add_argument('files', type=str, nargs='+', help='src, tgt, output')
    args = parser.parse_args()
    src_mat = sio.loadmat(args.files[0])
    tgt_mat = sio.loadmat(args.files[1])
    src_pose = src_mat['pose']
    tgt_pose = tgt_mat['pose']
    src = src_mat['vertex']
    tgt = tgt_mat['vertex']
    Tij = tgt_pose.dot(inverse(src_pose))
    src_pc = open3d.PointCloud()
    src_pc.points = open3d.Vector3dVector(src.T)
    tgt_pc = open3d.PointCloud()
    tgt_pc.points = open3d.Vector3dVector(tgt.T)
    #source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #        prepare_dataset(voxel_size)
    #draw_registration_result(src_pc, tgt_pc, Tij)
    open3d.estimate_normals(src_pc, search_param = open3d.KDTreeSearchParamHybrid(
            radius = 0.2, max_nn = 60))
    open3d.estimate_normals(tgt_pc, search_param = open3d.KDTreeSearchParamHybrid(
            radius = 0.2, max_nn = 60))
	
    source_down, source_fpfh = preprocess_point_cloud(src_pc, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(tgt_pc, voxel_size)
    #result_ransac = execute_global_registration(source_down, target_down,
    #        source_fpfh, target_fpfh, voxel_size)
    result_fast = execute_fast_global_registration(source_down, target_down,
            source_fpfh, target_fpfh, voxel_size)
    #print(result_ransac)
    #draw_registration_result(source_down, target_down,
    #        result_fast.transformation)
    #result_icp = result_fast
    result_icp = refine_registration(src_pc, tgt_pc,
            source_fpfh, target_fpfh, voxel_size)
    aerr = angular_distance_np(Tij[np.newaxis, :3, :3], result_icp.transformation[np.newaxis, :3, :3]).sum()
    tree_tgt = NN(n_neighbors = 1, algorithm='kd_tree').fit(np.asarray(tgt_pc.points))
    distances, indices = tree_tgt.kneighbors(np.asarray(src_pc.points)) # [np, 1], [np, 1]
    idx = np.where(distances < 0.2)[0]
    if len(idx) == 0:
        sigma = 1000000000
    else:
        sigma = np.median(distances[idx])
    print('sigma= %f' % sigma) 

    terr = np.linalg.norm(Tij[np.newaxis, :3, 3] - result_icp.transformation[np.newaxis, :3, 3], 2)
    sio.savemat(args.files[2], mdict={'Tij': result_icp.transformation, 'src': args.files[0], 'tgt': args.files[1], 'sigma': sigma, 'aerr': aerr, 'terr': terr}, do_compression=True)
    print('src=%s, tgt=%s, sigma=%f, aerr = %f, terr = %f' % (args.files[0], args.files[1], sigma, aerr, terr))
    #print(result_icp)
    #draw_registration_result(src_pc, tgt_pc, result_icp.transformation)
    #draw_registration_result(src_pc, tgt_pc, Tij)
