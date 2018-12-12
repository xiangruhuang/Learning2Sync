import sys
sys.path.append('../../')
from util import env, Reader, sio
home = env()
import sys
sys.path.append('%s/TransformationSync/' % home)

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
from scipy.stats import ortho_group as so3
from TS import project_so
import time
import glob
import pathlib
import argparse

def inverse(T):
    R, t = decompose(T)
    invT = np.zeros((4, 4))
    invT[:3, :3] = R.T
    invT[:3, 3] = -R.T.dot(t)
    invT[3, 3] = 1
    return invT

def pack(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T

def decompose(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

def construct_laplacian(n, eps):
    L = np.zeros((n*3, n*3))
    R = np.array([np.eye(3) for i in range(n)])
    #print(R)
    #R = so3.rvs(3, size=n)
    #print(R.shape)
    label = []
    R_in = []
    for i in range(n):
        for j in range(i+1, n):
            if np.random.uniform() < 0.2:
                Rij = R[j].dot(R[i].T) + np.random.randn(3, 3) * 0.5
            else:
                Rij = R[j].dot(R[i].T) + np.random.randn(3, 3) * eps
            Rij = project_so(Rij)
            R_in.append(Rij)
            if np.linalg.norm(Rij - R[j].dot(R[i].T), 'fro') < 0.1:
                label.append(1)
            else:
                label.append(0)
            L[i*3:(i+1)*3, j*3:(j+1)*3] = -Rij.T
            #L[i*3:(i+1)*3, j*3:(j+1)*3] = -Rij
    L = L + L.T
    for i in range(n):
        L[i*3:(i+1)*3, i*3:(i+1)*3] = (n-1)*np.eye(3)
    #print(L)
    R_in = np.array(R_in)
    return L, R, R_in, np.array(label)

class Sync(object):
    def __init__(self, args):
        self.mask = None
        self.theta = None
        self.L_split = None
        self.debug_dict = {}
        self.__construct_forward_and_backward__(args)
        assert args.kmax > 1

    """
        compute weighted laplacian matrix, possibly with its gradient to theta
        Input:
        `weight`: [n choose 2]
        `dweight`: [n choose 2, d]
        
        Output:
        `weighted_L`: [3n, 3n]
        `dL`: [d, 3n, 3n]
    """
    def __weighted_laplacian__(self, weight, dweight=None):
        if self.debug:
            start_time = time.time()
        n = self.n
        nc2 = int(n * (n-1) // 2)
        assert self.L_template is not None
        
        #if self.mask is None:
        #    indices = []
        #    values = []
        #    to1d = np.zeros((n, n)).astype(np.int32)
        #    count = 0
        #    for i in range(n):
        #        for j in range(i+1, n):
        #            to1d[i, j] = count
        #            count += 1
        #    for i in range(n):
        #        for j in range(i+1, n):
        #            for t1 in range(3):
        #                for t2 in range(3):
        #                    indices.append([(i*3+t1)*n*3 + j*3+t2, to1d[i, j]]); values.append(1.0)
        #                    indices.append([(j*3+t2)*n*3 + i*3+t1, to1d[i, j]]); values.append(1.0)
        #                    indices.append([(i*3+t1)*n*3 + i*3+t2, to1d[i, j]]); values.append(1.0 / (n-1.0))
        #                    indices.append([(j*3+t1)*n*3 + j*3+t2, to1d[i, j]]); values.append(1.0 / (n-1.0))
        #    if self.dtype == tf.float64:
        #        values = np.array(values, dtype=np.float64)
        #    else:
        #        values = np.array(values, dtype=np.float32)
        #    self.mask = tf.SparseTensor(indices=indices, values = values, dense_shape=[n*3*n*3, nc2])
        #w_mat = tf.sparse_tensor_dense_matmul(self.mask, tf.expand_dims(weight, axis=-1))
        #w_mat = tf.reshape(w_mat, [n*3, n*3])
        #weighted_L = tf.multiply(self.L_template, w_mat)
        """ Compute Weighted Laplacian """
        W = []
        offset = 0
        for i in range(n):
            li = n-(i+1)
            Wi = []
            Wi.append(tf.zeros(shape=[i+1], dtype=self.dtype))
            Wi.append(weight[offset:offset+li])
            offset += li
            Wi = tf.concat(Wi, axis=0)
            W.append(Wi)
        W = tf.stack(W, axis=0)
        W = tf.transpose(W) + W
        diag = tf.reduce_sum(W, axis=0) / (1.0 * n)
        W = tf.linalg.set_diag(W, diag)
        id3 = tf.linalg.LinearOperatorFullMatrix(tf.ones(shape=[3, 3], dtype=self.dtype))
        W_op = tf.linalg.LinearOperatorFullMatrix(W)
        W_mask = tf.linalg.LinearOperatorKronecker([W_op, id3]).to_dense() # [3n, 3n]
        weighted_L = tf.multiply(self.L_template, W_mask)
        #weighted_L = tf.Print(weighted_L, [weighted_L[:3, :3]], summarize=9, message='weighted_L')
        if dweight is not None:
            dweight = tf.transpose(dweight)
            d = self.theta_dim # should be defined already
            offset = 0
            dW = []
            for i in range(n):
                li = n-(i+1)
                dWi = []
                dWi.append(tf.zeros(shape=[d, i+1], dtype=self.dtype))
                dWi.append(dweight[:, offset:offset+li])
                #dLi_right = weight[offset:offset+li]
                dWi = tf.concat(dWi, axis=-1)
                dW.append(dWi)
                offset += li
                #for j in range(i+1, n):
                #    dL[i][j].append()
            dW = tf.stack(dW, axis=1) # [d, n, n]
            dW = tf.transpose(dW, (0, 2, 1)) + dW
            diag = tf.reduce_sum(dW, axis=1) / (1.0 * n)
            dW = tf.linalg.set_diag(dW, diag)
            dW_op = tf.linalg.LinearOperatorFullMatrix(dW)
            dW_mask = tf.linalg.LinearOperatorKronecker([dW_op, id3]).to_dense() # [d, 3n, 3n]
            dL = tf.multiply(dW_mask, self.L_template)
            #self.empirical = tf.reshape(weighted_L, [-1])
            #self.predict = tf.transpose(tf.reshape(dL, [d, -1]))
        else:
            dL = None
        if self.debug:
            print('weighted laplacian took %ss' % (time.time() - start_time))
            start_time = time.time()
        return weighted_L, dL
        
    """
        Input
        `weight`: of shape [nc2]
        `dweight`: dweight_dtheta of shape [nc2, d]
        `last`: for the last layer, compare with R_gt, 
            for other layers, compare with R_in
        
        Output
        `Loss`: of shape [nc2]
        `dLoss_dtheta`: jacobian of size [nc2, d]
            [i, j] is dLoss(i) / dweight(j)
        
    """
    def __weight_to_loss__(self, weight, dweight=None, last=False):
        def tensor_mul43(a, b):
            assert len(a.shape.as_list()) == 4
            assert len(b.shape.as_list()) == 3
            res = []
            for i in range(3):
                res_i = []
                for j in range(3):
                    p = tf.reduce_sum(tf.multiply(a[:, :, i, :], b[:, j, :]), axis=-1)
                    res_i.append(p)
                res_i = tf.stack(res_i, axis=-1)
                res.append(res_i)
            res = tf.stack(res, axis=-2)
            return res
        
        def tensor_mul33(a, b):
            assert len(a.shape.as_list()) == 3
            assert len(b.shape.as_list()) == 3
            res = []
            for i in range(3):
                res_i = []
                for j in range(3):
                    p = tf.reduce_sum(tf.multiply(a[:, i, :], b[:, j, :]), axis=-1)
                    res_i.append(p)
                res_i = tf.stack(res_i, axis=-1)
                res.append(res_i)
            res = tf.stack(res, axis=-2)
            return res
        
        def tensor_mul34(a, b):
            assert len(a.shape.as_list()) == 3
            assert len(b.shape.as_list()) == 4
            res = []
            for i in range(3):
                res_i = []
                for j in range(3):
                    p = tf.reduce_sum(tf.multiply(a[:, i, :], b[:, :, j, :]), axis=-1)
                    res_i.append(p)
                res_i = tf.stack(res_i, axis=-1)
                res.append(res_i)
            res = tf.stack(res, axis=-2)
            return res

        def tensor_mul42(a, b):
            assert len(a.shape.as_list()) == 4
            assert len(b.shape.as_list()) == 2
            res = []
            p0 = tf.multiply(a[:, :, :, 0], b[:, 0:1])
            p1 = tf.multiply(a[:, :, :, 1], b[:, 1:2])
            p2 = tf.multiply(a[:, :, :, 2], b[:, 2:3])
            res = p1 + p2 + p0
            return res
        if self.debug:
            start_time = time.time()
         
        n = self.n
        nc2 = int(n * (n-1) // 2)
        assert self.L_template is not None
        assert self.R_gt is not None
        assert self.R_in is not None
        #if debug:
        #    weight = tf.Print(weight, [weight], "weight", summarize=250)
        weighted_L, dL = self.__weighted_laplacian__(weight, dweight)
        #if debug:
        #    weighted_L = tf.Print(weighted_L, [weighted_L], "weighted_L", summarize=250)
        lamb, w = tf.linalg.eigh(weighted_L)
        #if debug:
        #    lamb = tf.Print(lamb, [lamb], "eigh")
        eigengap = lamb[3] - lamb[2]
        itr = 0
        while self.debug_dict.get('eigengap%d' % itr, None) is not None:
            itr += 1
        self.debug_dict['eigengap%d' % itr] = eigengap
        
        c = tf.sign(w[0, :] + 1e-12)
        w = tf.multiply(w, c)
        c = tf.sign(tf.linalg.det(w[:3, :3]))
        w = w * c
        A = tf.reshape(w[:, :3] * c, [n, 3, 3])
        S, U, V = tf.linalg.svd(A, full_matrices=True)
        R = tf.matmul(U, V, transpose_b = True)
        det = tf.linalg.det(R)
        R = tf.multiply(R, tf.expand_dims(tf.expand_dims(det, axis=-1), axis=-1)) 

        if not hasattr(self, 'indices_left'):
            indices_left = []
            indices_right = []
            for i in range(n):
                for j in range(i+1, n):
                    indices_left.append([j])
                    indices_right.append([i])
            indices_left = tf.constant(indices_left, dtype=tf.int32)
            indices_right = tf.constant(indices_right, dtype=tf.int32)
            self.indices_left = indices_left
            self.indices_right = indices_right
        indices_left = self.indices_left
        indices_right = self.indices_right
        
        R = tf.stack(R)
        if last:
            R_comp = self.R_gt
            R_comp_left = tf.gather_nd(R_comp, indices_left)
            R_comp_right = tf.gather_nd(R_comp, indices_right)
            R_comp = tensor_mul33(R_comp_left, R_comp_right)
        else:
            R_comp = self.R_in
        
        R_left = tf.gather_nd(R, indices_left)
        R_right = tf.gather_nd(R, indices_right)
        R_pairwise = tensor_mul33(R_left, R_right)
        
        loss = []
        counter2 = 0
        loss = 0.5*tf.reduce_sum(tf.square(R_pairwise - R_comp), axis=[1, 2])
        #for i in range(n):
        #    for j in range(i+1, n):
        #        Rij = tf.matmul(R[j], R[i], transpose_b=True)
        #        if not last:
        #            Rij_comp = R_comp[counter2, :, :]
        #        else:
        #            Rij_comp = tf.matmul(R_comp[j], R_comp[i], transpose_b=True)
        #        counter2 += 1
        #        loss.append(tf.nn.l2_loss(Rij - Rij_comp))
        #loss = tf.stack(loss)
        if self.debug:
            print('weight to loss forward took %s' % (time.time() - start_time))
            start_time = time.time()

        if dL is None:
            return loss, R 
        
        """ Compute Jacobian """

        inner_dim = self.theta_dim # should be defined already
        
        #dL = tf.sparse_reshape(dL, [nc2* 3*n, 3*n])

        #dL = tf.transpose(Jacob, (2, 0, 1))

        w_rest = w[:, 3:]
        lamb_rest = lamb[3:]

        #dAouter = []
        #for j in range(3):
        #    lamb_j = tf.diag(1.0/(lamb[j] - lamb_rest))
        #    dAouter_j = tf.matmul(w_rest, lamb_j)
        #    dAouter_j = tf.matmul(dAouter_j, w_rest, transpose_b=True)
        #    dAouter_j = tf.tensordot(dAouter_j, dL, axes=([1], [1]))
        #    dAouter_j = tf.tensordot(dAouter_j, w[:, j], axes=([2], [0]))
        #    dAouter_j = tf.transpose(dAouter_j)
        #    dAouter.append(dAouter_j)
        #dAouter = tf.stack(dAouter, axis=-1)
                   
        """ From Right Side """
        dAouter = []
        for j in range(3):
            right = tf.tensordot(dL, w[:, j], axes=([2], [0])) # [nc2, 3n] = [nc2, 3n, 3n] X [3n] 
            right = tf.matmul(w_rest, right, transpose_a=True, transpose_b=True, b_is_sparse=True) # [3n-3, nc2] = [3n, 3n-3].T X [nc2, 3n].T
            left = tf.multiply(w_rest, 1.0/(lamb[j] - lamb_rest)) # [3n, 3n-3] = [3n, 3n-3] (elementwise) [3n-3]
            dAouter_j = tf.matmul(left, right) # [3n, nc2]
            
            #lamb_j = tf.diag(1.0/(lamb[j] - lamb_rest))
            #dAouter_j = tf.matmul(w_rest, lamb_j)
            #dAouter_j = tf.matmul(dAouter_j, w_rest, transpose_b=True)
            #dAouter_j = tf.tensordot(dAouter_j, dL, axes=([1], [1]))
            #dAouter_j = tf.tensordot(dAouter_j, w[:, j], axes=([2], [0]))
            dAouter_j = tf.transpose(dAouter_j)
            dAouter.append(dAouter_j)
        dAouter = tf.stack(dAouter, axis=-1) # [nc2, 3n, 3]
        dAouter = tf.reshape(dAouter, [inner_dim, n, 3, 3])
        if self.debug:
            print('weight to loss backward part I took %s' % (time.time() - start_time))
            start_time = time.time()

         
        """ Sparse """
        #######################################
        #start_time = time.time()
        #dL_list = tf.unstack(dL, axis=0)
        #dAouter = []
        #for nc in range(nc2):
        #    dL_nc = dL_list[nc] # [3n, 3n]
        #    print(nc, time.time() - start_time)
        #    
        #    dAouter_nc = []
        #    for j in range(3):
        #        right = tf.matmul(dL_nc, w[:, j:(j+1)]) # [3n, 1] = [3n, 3n], [3n, 1]
        #        lamb_j = tf.diag(1.0/(lamb[j] - lamb_rest)) # [3n-3, 3n-3]
        #        dAouter_j = tf.matmul(w_rest, lamb_j) # [3n, 3n-3] = [3n, 3n-3] [3n-3, 3n-3]
        #        dAouter_j = tf.matmul(dAouter_j, w_rest, transpose_b=True) # [3n, 3n] = [3n, 3n-3] [3n, 3n-3].T
        #        dAouter_j = tf.matmul(dAouter_j, right) # [3n, 1] = [3n, 3n] [3n, 1]
        #        
        #        dAouter_nc.append(dAouter_j)
        #    dAouter_nc = tf.concat(dAouter_nc, axis=-1) # [3n, 3]
        #    dAouter.append(dAouter_nc)
        #dAouter = tf.stack(dAouter, axis=0) # [nc2, 3n, 3]

        ########################################

        dRouter = []
        for s in range(3):
            for t in range(3):
                if s == t:
                    continue
                dott = tensor_mul42(dAouter, V[:, :, t]) # [d, n, 3]
                dotst = tf.reduce_sum(tf.multiply(dott, U[:, :, s]), axis=-1)
                dots = tensor_mul42(dAouter, V[:, :, s]) # [d, n, 3]
                dotts = tf.reduce_sum(tf.multiply(dots, U[:, :, t]), axis=-1)
                
                scalar = tf.divide(dotst - dotts, S[:, s] + S[:, t]) # [d, n]
                scalar = tf.expand_dims(scalar, axis=-1) # [d, n, 1]
                scalar = tf.expand_dims(scalar, axis=-1) # [d, n, 1, 1]
                uv = tf.multiply(tf.expand_dims(U[:, :, s], axis=-1), tf.expand_dims(V[:, :, t], axis=-2)) # [n, 3, 3]
                res_st = tf.multiply(scalar, uv) # [d, n, 3, 3]
                dRouter.append(res_st)
        dRouter = tf.reduce_sum(tf.stack(dRouter, axis=0), axis=0) # [d, n, 3, 3]
        dRouter = tf.transpose(dRouter, (1, 0, 2, 3))
        
        #Uflat = tf.reshape(tf.transpose(U, (0, 2, 1)), [n, -1, 1])
        #Vflat = tf.reshape(tf.transpose(V, (0, 2, 1)), [n, 1, -1])
        #UVouter = tf.matmul(Uflat, Vflat) # [n, 9, 9]
        #
        #Souter = tf.expand_dims(S, axis=1) + tf.expand_dims(S, axis=2) # [n, 3, 3]
        #dRouter = []
        ##for s in range(3):
        ##    for t in range(3):
        ##        dRouter.append(Souter[:, s, t])
        #        
        #for i in range(n):
        #    #dRouter_i = tf.zeros(shape=[131, 3, 3], dtype=self.dtype)
        #    dRouter_i = []
        #    Ui = U[i]; Vi = V[i]; Si = S[i]
        #    dAouter_i = dAouter[:, i, :, :] # [nc2, 3, 3]
        #    #dotv = tf.tensordot(dAouter_i, Vi, axes=([2], [0])) # [nc2, 3, 3]
        #    #dotu = tf.tensordot(dotv, Ui, axes=([1], [0])) # [nc2, 3, 3]
        #    #symmetric = tf.transpose(dotu, (0, 2, 1)) - dotu # [nc2, 3, 3]
        #    #scalar = tf.divide(symmetric, Souter[i, :, :]) # [nc2, 3, 3]
        #    for s in range(3):
        #        for t in range(3):
        #            if s == t:
        #                continue
        #            
        #            dott = tensor_mul42(dAouter, V[:, :, t]) # [d, n, 3]
        #            dotst = tf.reduce_sum(tf.multiply(dott, U[:, s, :]), axis=-1) # [d, n]
        #            
        #            #scalar1 = tf.tensordot(tf.tensordot(Ui[:, s], dAouter_i, axes=([0], [1])), Vi[:, t], axes=([1], [0]))
        #            #scalar1 = tf.tensordot(tf.tensordot(dAouter_i, Vi[:, t], axes=([2], [0])), Ui[:, s], axes=([1], [0]))
        #            scalar1 = tf.tensordot(dott[:, i, :], Ui[:, s], axes=([1], [0]))
        #            scalar2 = tf.tensordot(tf.tensordot(Ui[:, t], dAouter_i, axes=([0], [1])), Vi[:, s], axes=([1], [0]))
        #            print(scalar1.shape, scalar2.shape)
        #            scalar = scalar1 - scalar2
        #            #scalar = symmetric[:, s, t]
        #            scalar = scalar / (Souter[i, s, t])
        #            #uv = tf.matmul(tf.expand_dims(Ui[:, s], axis=-1), tf.expand_dims(Vi[:, t], axis=0))
        #            #uv = tf.expand_dims(uv, axis=0)
        #            uv = UVouter[i:(i+1), s*3:(s+1)*3, t*3:(t+1)*3]
        #            scalar = tf.expand_dims(scalar, axis=-1)
        #            temp = tf.tensordot(scalar, uv, axes=([-1], [0]))
        #            dRouter_i.append(temp)
        #    dRouter_i = tf.reduce_sum(tf.stack(dRouter_i), axis=0)
        #    dRouter.append(dRouter_i)
        #
        #dRouter = tf.stack(dRouter, axis=0)
        if self.debug:
            print('weight to loss backward part II took %s' % (time.time() - start_time))
            start_time = time.time()
        
        dRouter_left = tf.gather_nd(dRouter, indices_left)
        dRouter_right = tf.gather_nd(dRouter, indices_right)

        dRouter_left = tf.transpose(dRouter_left, (1, 0, 2, 3))
        dRouter_right = tf.transpose(dRouter_right, (1, 0, 2, 3))
        dR_pairwise = tensor_mul43(dRouter_left, R_right) + tensor_mul34(R_left, dRouter_right)
        grad = tf.reduce_sum(tf.multiply(dR_pairwise, R_pairwise - R_comp), axis=[-2, -1])
        grad = tf.transpose(grad)
        #grad = tf.zeros(shape=[inner_dim], dtype=self.dtype)
        #grad = []
        #counter2 = 0
        #for i in range(n):
        #    for j in range(i+1, n):
        #        #dRij = tf.tensordot(dRouter[j], R[i], axes=([2], [1]))
        #        #dRij = dRij + tf.transpose(tf.tensordot(R[j], dRouter[i], axes=([1], [2])), (1,0,2))
        #        dRij = dR_pairwise[:, counter2, :, :]
        #        #Rij = tf.matmul(R[j], R[i], transpose_b=True)
        #        Rij = R_pairwise[counter2, :, :]
        #        Rij_comp = R_comp[counter2, :, :]
        #        counter2 += 1
        #        #Rij_gt = tf.matmul(R_gt[j], R_gt[i], transpose_b=True)
        #        dLoss = tf.tensordot(dRij, Rij - Rij_comp, axes=([1, 2], [0, 1]))
        #        grad.append(dLoss)
        #grad = tf.stack(grad, axis=0)
        
                
        if self.debug:
            print('weight to loss backward part III took %s' % (time.time() - start_time))
            start_time = time.time()

        return loss, R, grad
       
    """
        Input:
        `loss`:
        
        Output
        `weight`: weight(loss, theta) [nc2]
        `Jweight_loss`: [nc2, nc2]
        `Jweight_theta`: [nc2, d]
    """
    def __loss_to_weight__(self, loss):
        if self.debug:
            start_time = time.time()
        n = self.n
        nc2 = int(n * (n-1) // 2)
        if self.theta is None:
            self.theta_dim = 131
            self.theta = tf.get_variable(name='theta', shape=[self.theta_dim], dtype=self.dtype, initializer = tf.constant_initializer(1.0, dtype=self.dtype))
        d = self.theta_dim
        
        """ reweighted version """
        #feats = tf.concat([tf.expand_dims(loss, axis=-1), self.feat], axis=-1) # [nc2, 129]
        dp = tf.reduce_sum(tf.multiply(self.feat, self.theta[3:]), axis=1) + self.theta[1]
        sigmoid = tf.sigmoid(dp)
        mloss = tf.multiply(loss, sigmoid)
        exp_theta2 = tf.exp(self.theta[2] * self.theta[0])
        ploss = tf.pow(mloss, self.theta[0])
        
        weight = exp_theta2 / (exp_theta2 + ploss)
        weight_out = weight * self.weight0
        
        itr = 0
        while self.debug_dict.get('ploss%d' % itr, None) is not None:
            itr += 1
        self.debug_dict['ploss%d' % itr] = mloss
        self.debug_dict['exp%d' % itr] = exp_theta2
        
        """ Derivatives """
        dweight_out_dweight = self.weight0
        dweight_dexp_theta2 = 1.0 / (exp_theta2 + ploss) - exp_theta2 / tf.square(exp_theta2 + ploss) # [nc2]
        
        dweight_dploss = -exp_theta2 / tf.square(exp_theta2 + ploss) # [nc2]
        
        dploss_dmloss = self.theta[0] * tf.pow(mloss, self.theta[0] - 1.0) #[nc2]
        dploss_dtheta0 = ploss * tf.log(mloss) #[nc2]
        
        dmloss_dloss = sigmoid # [nc2]
        dmloss_dsigmoid = loss # [nc2]
        
        dsigmoid_ddp = sigmoid * (1.0-sigmoid) # [nc2]
        dsigmoid_dtheta1 = dsigmoid_ddp # [1]
        ddp_dthetarest = self.feat # [nc2, 128]
        
        dexp_theta2_dtheta2 = self.theta[0] * exp_theta2 # [nc2]
        dexp_theta2_dtheta0 = self.theta[2] * exp_theta2 # [nc2]
        
        dweight_dtheta0 = tf.expand_dims(dweight_dexp_theta2 * dexp_theta2_dtheta0 + dweight_dploss * dploss_dtheta0, axis=-1) # [nc2, 1]
        dweight_dtheta1 = tf.expand_dims(dweight_dploss * dploss_dmloss * dmloss_dsigmoid * dsigmoid_dtheta1, axis=-1) # [nc2, 1]
        dweight_dtheta2 = tf.expand_dims(dweight_dexp_theta2 * dexp_theta2_dtheta2, axis=-1) # [nc2, 1]
        dweight_dthetarest = tf.multiply(tf.expand_dims(dweight_dploss * dploss_dmloss * dmloss_dsigmoid * dsigmoid_ddp, axis=-1), ddp_dthetarest) # [nc2, 128]
        
        dweight_dtheta = tf.concat([dweight_dtheta0, dweight_dtheta1, dweight_dtheta2, dweight_dthetarest], axis=1) # [nc2, 131]
        dweight_out_dtheta = tf.multiply(tf.expand_dims(self.weight0, axis=-1), dweight_dtheta) # [nc2, 131]
        
        dweight_dloss = tf.expand_dims(dweight_dploss * dploss_dmloss * dmloss_dloss, axis=-1)
        dweight_out_dloss = tf.multiply(tf.expand_dims(self.weight0, axis=-1), dweight_dloss)
        if self.debug:
            print('loss to weight took %s' % (time.time() - start_time))
            start_time = time.time()
        return weight_out, dweight_out_dloss, dweight_out_dtheta
    """
        Input
        `loss`: shape is [nc2]
        
        Output:
        `feature`: shape is [nc2 * d]
        `Jfeature_loss`: jacobian of size [nc2 * d, nc2]
            [i, k] is dfeature(i) / dloss(k)
    """
    def __loss_to_feature__(self, loss, compute_gradient=True):
        nc2 = self.nc2
        d = self.theta_dim
        loss2 = tf.square(loss)
        #loss_inv = 1.0/loss
        feature = [loss, loss2, tf.ones_like(loss, dtype=self.dtype)]
        feature = tf.stack(feature, axis=-1)
        feature = tf.reshape(feature, [-1])
        if not compute_gradient:
            return feature
        assert feature.shape[0] == self.nc2 * self.theta_dim
        Jacob = tf.constant(0.0, shape=[nc2, d, nc2], dtype=self.dtype)
        
        grad0 = tf.diag(tf.constant(1.0, shape=[nc2], dtype=self.dtype))
        grad1 = tf.diag(loss * 2.0)
        grad2 = tf.diag(tf.zeros_like(loss, dtype=self.dtype))
        #grad2 = tf.diag(-1.0/tf.square(loss))
        grads = [grad0, grad1, grad2]

        grad = tf.concat(grads, axis=0)
        assert grad.shape==[nc2*d, nc2]

        return feature, grad
 
    """
        Input
        `feature`: shape is [nc2 * d]
        
        Output:
        `weight`: shape is [nc2]
        `Jweight_feature`: jacobian of size [nc2, nc2 * d]
            [i, k] is dweight(i) / dfeature(k)
    """
    def __feature_to_weight__(self, feature, compute_gradient=True):
        d = self.theta_dim
        nc2 = self.nc2
        #if debug:
        #    feature = tf.Print(feature, [feature], "feature, feature_to_weight\n", summarize=250)
        feature_mat = tf.reshape(feature, [nc2, d])
        weight = tf.matmul(feature_mat, tf.expand_dims(self.theta, axis=-1))
        weight = tf.squeeze(weight, 1)
        assert weight.shape.as_list()[0] == nc2
        if not compute_gradient:
            return weight
        indices = []
        values = tf.concat([self.theta] * nc2, axis=0)

        for i in range(nc2):
            for j in range(d):
                indices.append([i, i*d+j])
        Jacob = tf.SparseTensor(indices = indices, values = values, dense_shape=[nc2, nc2*d])
        Jacob = tf.sparse_tensor_to_dense(Jacob)
        assert Jacob.shape.as_list() == [nc2, nc2*d]
        return weight, Jacob
        

    def __multiply_and_sigmoid__(self, weight, weight0):
        sigmoid_weight = tf.sigmoid(tf.multiply(weight, weight0))
        #if debug:
        #    sigmoid_weight = tf.Print(sigmoid_weight, [sigmoid_weight], "sigmoid_weight\n")
        Jacob = tf.diag(tf.multiply(tf.multiply(sigmoid_weight, 1.0 - sigmoid_weight), weight0))
        return sigmoid_weight, Jacob
        

    def __construct_forward_and_backward__(self, args):
        self.n = args.n # dimension
        self.dtype = args.dtype
        self.debug = args.debug
        n = self.n
        nc2 = int(n*(n-1)//2) # for convenience
        kmax = args.kmax
        learning_rate = args.learning_rate
        self.R_gt = tf.placeholder(shape=[n, 3, 3], dtype=self.dtype)
        self.R_in = tf.placeholder(shape=[nc2, 3, 3], dtype=self.dtype)
        self.L_template = tf.placeholder(shape=[3*n, 3*n], dtype=self.dtype) # L template
        self.feat = tf.placeholder(shape=[nc2, 128], dtype=self.dtype)
        
        self.weight0 = tf.placeholder(shape=[nc2], dtype=self.dtype)
        weight0 = self.weight0
        #weight0 = tf.Print(weight0, [weight0], 'weight0', summarize=1000)
        #if debug:
        #    weight0 = tf.Print(weight0, [weight0], "weight0\n")
        loss0, R0 = self.__weight_to_loss__(weight0, dweight = None) # shape: [nc2]
        #if debug:
        #    loss0 = tf.Print(loss0, [loss0], "loss0\n")
        self.losses = [loss0]
        self.Rlist = [R0]
        self.weights = [weight0]
        for k in range(1, kmax):
            with tf.name_scope('iter%d' % k):
                print('building iteration = %d' % k)
                if k == 1:
                    #feature_k = self.__loss_to_feature__(self.losses[k-1], compute_gradient=False)
                    #if debug:
                    #    feature_k = tf.Print(feature_k, [feature_k], "feature_k, k=%d\n" % k, summarize=250)
                    #weight_k = self.__feature_to_weight__(feature_k, compute_gradient=False)
                    #Jacob = tf.reshape(feature_k, [nc2, d])
                    #if debug:
                    #    weight_k = tf.Print(weight_k, [weight_k], "weight_k, k=%d\n" % k)
                    #weight_k, Jsigmoid = self.__multiply_and_sigmoid__(weight_k, weight0)
                    #Jacob = tf.matmul(Jsigmoid, Jacob)
                    #if debug:
                    #    Jacob = tf.Print(Jacob, [Jacob], "Jacob, k=%d\n" % k)
                    weight_k, Jweight_loss, Jacob = self.__loss_to_weight__(self.losses[k-1])
                    self.weights.append(weight_k)
                else:
                    #feature_k, Jloss_feature = self.__loss_to_feature__(self.losses[k-1])
                    #if debug:
                    #    feature_k = tf.Print(feature_k, [feature_k], "feature_k, k=%d\n" % k)
                    #Jacob = tf.matmul(Jloss_feature, Jacob)
                    #if debug:
                    #    Jacob = tf.Print(Jacob, [Jacob], "Jacob, k=%d, 1\n" % k)
                    #weight_k, Jweight_feature = self.__feature_to_weight__(feature_k)
                    #Jacob = tf.matmul(Jweight_feature, Jacob) + tf.reshape(feature_k, [nc2, d])
                    #if debug: 
                    #    weight_k = tf.Print(weight_k, [weight_k], "weight_k, k=%d\n" % k)
                    #weight_k, Jsigmoid = self.__multiply_and_sigmoid__(weight_k, weight0)
                    #Jacob = tf.matmul(Jsigmoid, Jacob)
                    #if debug: 
                    #    Jacob = tf.Print(Jacob, [Jacob], "Jacob, k=%d, 2\n" % k)
                    weight_k, Jweight_loss, Jweight_theta = self.__loss_to_weight__(self.losses[k-1])
                    #print(Jweight_loss.shape, Jweight_theta.shape)
                    Jacob = tf.multiply(Jweight_loss, Jacob) + Jweight_theta
                    self.weights.append(weight_k)
                if k == kmax-1:
                    loss_k, Rk, Jacob = self.__weight_to_loss__(weight_k, dweight=Jacob, last=True)
                else:
                    loss_k, Rk, Jacob = self.__weight_to_loss__(weight_k, dweight=Jacob, last=False)
                #if k == 1:
                #    self.predict = Jacob
                #    self.empirical = loss_k
                #if debug:
                #    loss_k = tf.Print(loss_k, [loss_k], "loss_k, k=%d\n" % k)
                #Jacob = tf.matmul(Jloss_weight, Jacob)
                self.losses.append(loss_k)
                self.Rlist.append(Rk)
        
        self.loss = tf.reduce_sum(self.losses[kmax-1])
        #if debug: 
        #    self.loss = tf.Print(self.loss, [self.loss], "self.loss")
        self.loss_mat = tf.stack(self.losses, axis=0)
        self.weight_mat = tf.stack(self.weights, axis=0)
        self.grad = tf.reduce_sum(Jacob, axis=0)
        self.learning_rate = tf.get_variable(name='learning_rate', shape=[], dtype=self.dtype, initializer=tf.constant_initializer(learning_rate, dtype=self.dtype))
        self.train_op = self.theta.assign_add(-1.0*self.learning_rate*self.grad)
        self.R = self.Rlist[-1]
        self.predict = self.grad
        self.empirical = self.loss

    def feed_dict(self):
        d = {}
        d['R_gt'] = self.R_gt
        d['R_in'] = self.R_in
        d['L'] = self.L_template
        d['w'] = self.weight0
        d['feat'] = self.feat
        return d

    def eval_dict(self):
        d = {}
        d['R'] = self.R
        d['loss'] = self.loss
        d['theta'] = self.theta
        return d  

    def train_dict(self):
        d = {}
        d['R'] = self.R
        d['loss'] = self.loss
        d['grad'] = self.grad
        #d['train_op'] = self.train_op
        d['loss_mat'] = self.loss_mat
        d['weight_mat'] = self.weight_mat
        d['theta'] = self.theta
        return d
    
    def get_debug_dict(self):
        self.debug_dict['predict'] = self.predict
        self.debug_dict['empirical'] = self.empirical
        return self.debug_dict
    
    def __dump__(self, v):
        print(v.name, type(v), v.shape)

    def line_search(self, sess, c1, c2, feed_dict):
        if not hasattr(self, 'assign_val'):
            self.assign_val = tf.placeholder(shape=[], dtype=self.dtype)
            self.assign_op = self.learning_rate.assign(self.assign_val)
            self.perturb_val = tf.placeholder(shape=self.theta.shape, dtype=self.dtype)
            self.perturb_op = self.theta.assign_add(self.perturb_val)

        assign_val = self.assign_val
        assign_op = self.assign_op
        perturb_val = self.perturb_val
        perturb_op = self.perturb_op
        
        
        f0, grad, x = sess.run([self.loss, self.grad, self.theta], feed_dict=feed_dict)
        lr_val = 1e-2
        #while (x - lr_val * grad < 0).any():
        #    lr_val = lr_val * 0.9
        #sess.run(assign_op, feed_dict={assign_val:lr_val})
        p = -grad
        sess.run(perturb_op, feed_dict={perturb_val: p * lr_val})
        fp, gradp = sess.run([self.loss, self.grad], feed_dict=feed_dict)
        #print('', end="") 
        while (lr_val > 1e-5) and ((fp > f0 + c1 * lr_val * p.dot(grad)) or (-p.dot(gradp) > -c2 * p.dot(grad))):
            #print('lr:', lr_val)
            #print('1:', fp, f0 +  c1 * lr_val * p.dot(grad))
            #print('2:', -p.dot(gradp), -c2 * p.dot(grad))
            #print('\rlearning rate = %f' % lr_val, end="")
            sess.run(perturb_op, feed_dict={perturb_val: -p * lr_val})
            lr_val = lr_val * 0.9
            #sess.run(assign_op, feed_dict={assign_val:lr_val})
            
            sess.run(perturb_op, feed_dict={perturb_val: p * lr_val})
            fp, gradp = sess.run([self.loss, self.grad], feed_dict=feed_dict)
        if (lr_val < 1e-5):
            """ Jump """
            print('jumping...')
            sess.run(perturb_op, feed_dict={perturb_val: -p * lr_val})
            self.jump(sess, feed_dict=feed_dict)
        else:
            print('learning_rate=%f' % lr_val) 
        #print('lr:', lr_val)
        #print('1:', fp, f0 +  c1 * lr_val * p.dot(grad))
        #print('2:', -p.dot(gradp), -c2 * p.dot(grad))
        #print('\rlearning_rate = %f' % lr_val)
    
    def jump(self, sess, width1 = 0.1, width2 = 0.1, feed_dict = None):
        if not hasattr(self, 'assign_val'):
            self.assign_val = tf.placeholder(shape=[], dtype=self.dtype)
            self.assign_op = self.learning_rate.assign(self.assign_val)
            self.perturb_val = tf.placeholder(shape=self.theta.shape, dtype=self.dtype)
            self.perturb_op = self.theta.assign_add(self.perturb_val)

        assign_val = self.assign_val
        assign_op = self.assign_op
        perturb_val = self.perturb_val
        perturb_op = self.perturb_op
        theta = sess.run(self.theta)
        N = 10
        m = []
        X = []
        Y = []
        min_loss = 1000000000.0
        min_dev = None
        for i in range(-N, N+1):
            dev1 = width1 * 1.0 / N * i
            for j in range(-N, N+1):
                dev2 = width2 * 1.0 / N * j
                dev = np.array([dev1, dev2])
                sess.run(perturb_op, feed_dict={perturb_val: dev})
                loss_ij = sess.run(model.loss, feed_dict=feed_dict)
                if loss_ij < min_loss:
                    min_loss = loss_ij
                    min_dev = dev
                sess.run(perturb_op, feed_dict={perturb_val: -dev})
        sess.run(perturb_op, feed_dict={perturb_val: min_dev})

    def plot_local(self, sess, width1 = 0.1, width2 = 0.05, feed_dict = None):
        if not hasattr(self, 'assign_val'):
            self.assign_val = tf.placeholder(shape=[], dtype=self.dtype)
            self.assign_op = self.learning_rate.assign(self.assign_val)
            self.perturb_val = tf.placeholder(shape=self.theta.shape, dtype=self.dtype)
            self.perturb_op = self.theta.assign_add(self.perturb_val)

        assign_val = self.assign_val
        assign_op = self.assign_op
        perturb_val = self.perturb_val
        perturb_op = self.perturb_op
        theta = sess.run(self.theta)
        N = 10
        m = []
        X = []
        Y = []
        for i in range(-N, N+1):
            mi = []
            Xi = []
            Yi = []
            dev1 = width1 * 1.0 / N * i
            for j in range(-N, N+1):
                dev2 = width2 * 1.0 / N * j
                Xi.append(theta[0] + dev1)
                Yi.append(theta[1] + dev2)
                dev = np.array([dev1, dev2])
                sess.run(perturb_op, feed_dict={perturb_val: dev})
                loss_ij = sess.run(model.loss, feed_dict=feed_dict)
                sess.run(perturb_op, feed_dict={perturb_val: -dev})
                mi.append(loss_ij)
            m.append(mi)
            X.append(Xi)
            Y.append(Yi)
        m = np.array(m)
        X = np.array(X)
        Y = np.array(Y)
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        #import ipdb; ipdb.set_trace() 
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        
        ax.plot_wireframe(X, Y, m, rstride=1, cstride=1) 
        
        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.show()
     

def run_test(sess, model, feed_dict=None):
    fdict = model.feed_dict()
    train_dict = model.train_dict()
    debug_dict = model.get_debug_dict()
    n = model.n
    nc2 = int(n*(n-1)//2)
    w0 = np.ones(nc2)
    np.set_printoptions(precision=16)
    eps = 1e-6
    for ii in range(1):
        if feed_dict is None:
            L, R_gt, R_in, _ = construct_laplacian(n, eps=0.01)
            feed_dict = {
                fdict['R_gt']: R_gt,
                fdict['L']: L,
                fdict['w']: w0,
                fdict['R_in']: R_in,
            }
        res = sess.run(debug_dict, feed_dict=feed_dict)
        empirical0 = res['empirical']
        grad = res['predict']
        for t in range(model.theta_dim):
            delta_theta = np.zeros(model.theta_dim)
            delta_theta[t] = 1.0
            sess.run(model.theta.assign_add(delta_theta * eps))
            res = sess.run(debug_dict, feed_dict=feed_dict)
            empirical = res['empirical']
            sess.run(model.theta.assign_add(-delta_theta * eps))
            delta = empirical - empirical0
            err = grad.dot(delta_theta) - delta / eps
            if isinstance(err, np.ndarray):
                err = np.linalg.norm(err, 2)
                delta = np.linalg.norm(delta, 2)
            print(delta / eps, err)

def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    n = R.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric

def angular_distance_pairwise(R, R_gt):
    n = R.shape[0]
    
    Rij = []; Rij_gt = [] 
    for i in range(n):
        for j in range(i+1, n):
            Rij.append(R[j].dot(R[i].T))
            Rij_gt.append(R_gt[j].dot(R_gt[i].T))
    Rij = np.array(Rij)
    Rij_gt = np.array(Rij_gt)
    return np.mean(angular_distance_np(Rij, Rij_gt))

def binary(w0):
    l = []
    for i in range(len(w0)):
        if w0[i] > 0.5:
            l.append(1.0)
        else:
            l.append(0.00001)
    return np.array(l)

def load_real(n, mat_file):
    mat = sio.loadmat(mat_file)
    #rescue = mat_file.replace('super4pcs', 'fgr')
    #rescue_mat = sio.loadmat(rescue)
    #Trescue = rescue_mat['Trel'][:4*n, :4*n]
    L = mat['L'][:3*n, :3*n]
    R_gt = mat['pose']
    R_gt = R_gt[:n, :3, :3]
    #Trel = sio.loadmat('../temp.mat')['fuck']
    Trel = mat['Trel'][:4*n, :4*n]
    R_in_vec = np.zeros((n*(n-1)//2, 3, 3))
    counter = 0
    label = mat['predict']
    label_vec = np.zeros(n*(n-1)//2)
    feat = mat['feat']
    feat_vec = np.zeros((n*(n-1)//2, 128))
    fc2weight = mat['weight'][0, :]
    fc2bias = mat['bias'][0, :]
    L = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(i+1, n):
            Tij = Trel[i*4:(i+1)*4, j*4:(j+1)*4]
            if abs(Tij[3, 3] - 1.0) > 0.1:
                Tij = Trel[j*4:(j+1)*4, i*4:(i+1)*4]
                assert abs(Tij[3, 3] - 1.0) < 0.1
                Tij = inverse(Tij)
            U, S, VT = np.linalg.svd(Tij[:3, :3])
            
            if (abs(S - 1.0) > 0.01).any():
                Tij[:3, :3] = U.dot(VT)
                if np.linalg.det(Tij[:3, :3]) < 0.0:
                    Tij[:3, 2] = -Tij[:3, 2]
            if (i+j) % 2 == 2:
                R_in_vec[counter, :, :] = R_gt[j].dot(R_gt[i].T)
                L[i*3:(i+1)*3, j*3:(j+1)*3] = -(R_gt[j].dot(R_gt[i].T)).T
                label_vec[counter] = 1.0
            else:
                R_in_vec[counter, :, :] = Tij[:3, :3]
                label_vec[counter] = label[i, j]
                L[i*3:(i+1)*3, j*3:(j+1)*3] = -Tij[:3, :3].T
            feat_vec[counter, :] = feat[i, j, :]
            counter += 1

    L = L.T + L
    for i in range(n):
        L[i*3:(i+1)*3, i*3:(i+1)*3] = (n-1.0) * np.eye(3)
    #import ipdb; ipdb.set_trace()

    return L, R_gt, R_in_vec, label_vec, feat_vec, fc2weight, fc2bias

def main():
    parser = argparse.ArgumentParser(description='Train Hyper-parameters for Iterative Reweighted Rot Sync')
    parser.add_argument('--dataset', type=str, help='"scannet" or "redwood", default to redwood', default='redwood')
    parser.add_argument('--source', type=str, help='Source of Input: "fgr" or "super4pcs", default to fgr', default='fgr')
    parser.add_argument('--debug', action='store_true', help='debug switch')
    parser.add_argument('--dtype', type=str, default='tf.float64', help='data type to use: default to tf.float64')
    parser.add_argument('--kmax', type=int, default=4, help='number of iterations to unfold, default to 4')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='learning rate, default to 0.1')
    parser.add_argument('--n', type=int, default=30, help='number of objects for each scene/model, default to 30')

    args = parser.parse_args()
    if args.dtype == 'tf.float64':
        args.dtype = tf.float64

    dataset = args.dataset
    source = args.source

    model = Sync(args)
    n = args.n
    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())
        fdict = model.feed_dict()

        train_dict = model.train_dict()
        debug_dict = model.get_debug_dict()
        eval_dict = model.eval_dict()
        with open('%s/%s.train_list' % (home, dataset), 'r') as fin:
            train_models = [('/').join(line.strip().split('/')[-2:]) for line in fin.readlines()]
            train_list = ['%s/%s.predicts/%s/%s.mat' % (home, dataset, source, model) for model in train_models]
        with open('%s/%s.test_list' % (home, dataset), 'r') as fin:
            test_models = [('/').join(line.strip().split('/')[-2:]) for line in fin.readlines()]
            test_list = ['%s/%s.predicts/%s/%s.mat' % (home, dataset, source, model) for model in test_models]
        print(test_list)
        last_epoch = sorted([int(save.split('/')[-1].split('.')[0])for save in glob.glob('%s_train/%s/*.mat' % (dataset, source))])
        if len(last_epoch) == 0:
            last_epoch = -1
            initialized = False
        else:
            last_epoch = last_epoch[-1]
            initialized = True
            theta_load = sio.loadmat('%s_train/%s/%d.mat' % (dataset, source, last_epoch))['theta'][0] 
            sess.run(model.theta.assign(theta_load))
            print('=> loaded %s_train/%s/%d.mat ...' % (dataset, source, last_epoch))

        for epoch in range(last_epoch+1, 10000):
            aerrs = []
            grad_sum = np.zeros(model.theta_dim)
            for mat_file in train_list:
                if not os.path.exists(mat_file):
                    continue
                shapeid = mat_file.split('/')[-1].split('.')[0]
                #print(mat_file)
                if not initialized:
                    L, R_gt, R_in, label, feat, fc2weight, fc2bias = load_real(n, mat_file)
                    theta0 = np.array([2.0])
                    #print(theta0.shape, fc2weight.shape, fc2bias.shape)
                    initial_theta = np.concatenate((theta0, -5*(fc2bias-0.5), [np.log(0.1)], -5*fc2weight), axis=0)
                    sess.run(model.theta.assign(initial_theta))
                else:
                    L, R_gt, R_in, label, feat, _, _ = load_real(n, mat_file)
                    
                
                #import ipdb; ipdb.set_trace()   
                counter = 0 
                idx_good = []; idx_bad = []
                for ii in range(n):
                    for j in range(ii+1, n):
                        if label[counter] < 0.2:
                            counter += 1
                            continue
                        Rij_gt = R_gt[j].dot(R_gt[ii].T)
                        aerr = angular_distance_np(Rij_gt[np.newaxis, :, :], R_in[counter][np.newaxis, :, :]).sum()
                        if aerr < 30.0:
                            idx_good.append(counter)
                        else:
                            idx_bad.append(counter)
                        counter += 1
                idx_good = np.array(idx_good).astype(np.int32)
                idx_bad = np.array(idx_bad).astype(np.int32)

                for f in feat:
                    assert np.linalg.norm(f, 2) > 1e-6
                #for i in range(len(label)):
                #    if label[i] > 0.5:
                #        label[i] = 1.0
                #    else:
                #        label[i] = 0.0
                w0 = np.clip(label, 0.0001, 1.0)
                #w0 = binary(w0)
                #w0 = w0 ** 8
                feed_dict2 = {
                    fdict['R_gt']: R_gt,
                    fdict['L']: L,
                    fdict['w']: w0,
                    fdict['R_in']: R_in,
                    fdict['feat']: feat,
                }
                np.set_printoptions(precision=6)
                np.set_printoptions(linewidth=160)
                    
                res = sess.run(train_dict, feed_dict=feed_dict2)
                grad_i = res['grad']
                if np.linalg.norm(grad_i, 2) > 1.0:
                    grad_i = grad_i / np.linalg.norm(grad_i, 2) * 1.0
                
                aerr = angular_distance_pairwise(res['R'], R_gt)
                if aerr < 1000.0:
                    aerrs.append(aerr)
                    grad_sum += grad_i
                    if model.debug:
                        print('\tshapeid=%s, grad=%s, loss=%s, theta=%s, mean angular distance=%f' % (shapeid, np.linalg.norm(res['grad']), res['loss'], res['theta'][:4], aerr))
                        debug = sess.run(debug_dict, feed_dict=feed_dict2)
                        for k in range(args.kmax):
                            print('eigengap%d=%f' % (k, debug['eigengap%d' % k]))
                        
                    #if epoch == 0:
                    #    debug = sess.run(debug_dict, feed_dict=feed_dict2)
                    #    for k in range(args.kmax):
                    #        print('k=%d' % k)
                    #        print('weight_good', np.mean(res['weight_mat'][k, idx_good]), len(idx_good))
                    #        print('weight_bad', np.mean(res['weight_mat'][k, idx_bad]), len(idx_bad))
                    #        print('loss_good', np.mean(res['loss_mat'][k, idx_good]))
                    #        print('loss_bad', np.mean(res['loss_mat'][k, idx_bad]))
                    #    for k in range(args.kmax-1):
                    #        print('ploss%d_good' % k, np.mean(debug['ploss%d' % k][idx_good]))
                    #        print('ploss%d_bad' % k, np.mean(debug['ploss%d' % k][idx_bad]))
                    #        print('exp%d' % k, debug['exp%d' % k])
                #if i % 100 == 0:
                #    import ipdb; ipdb.set_trace() 
                #print('loss_mat', res['loss_mat'])
                        #print(np.linalg.norm(debug['wL0'], 'fro'))
                    #model.plot_local(sess, feed_dict=feed_dict)
                #model.line_search(sess, 0.0001, 0.95, feed_dict)
                #print('weight_mat', res['weight_mat'][1, :])
                #print('theta', theta)
                #print('loss_mat', res['loss_mat'])
            
            sess.run(model.theta.assign_add(-args.learning_rate * grad_sum / len(train_list)))
            print('epoch %d, train: mean aerr = %f (out of %d)' % (epoch, np.mean(aerrs), len(aerrs)))
            aerrs = []
            for mat_file in test_list:
                if not os.path.exists(mat_file):
                    continue
                shapeid = mat_file.split('/')[-1].split('.')[0]
                L, R_gt, R_in, label, feat, _, _ = load_real(n, mat_file)
                w0 = np.clip(label, 0.0001, 1.0)
                #w0 = binary(w0)
                feed_dict2 = {
                    fdict['R_gt']: R_gt,
                    fdict['L']: L,
                    fdict['w']: w0,
                    fdict['R_in']: R_in,
                    fdict['feat']: feat
                }
                np.set_printoptions(precision=6)
                np.set_printoptions(linewidth=160)

                res = sess.run(eval_dict, feed_dict=feed_dict2)
                pathlib.Path('%s_test/%s' % (dataset, source)).mkdir(exist_ok=True, parents=True)
                sio.savemat('%s_test/%s/%s.mat' % (dataset, source, shapeid), mdict={'R': res['R']})
                aerr = angular_distance_pairwise(res['R'], R_gt)
                if aerr < 1000.0:
                    aerrs.append(aerr)
                    if model.debug:
                        print('\tshapeid=%s, loss=%s, mean angular distance=%f' % (shapeid, res['loss'], aerr))
                
            print('epoch %d, test: mean aerr = %f (out of %d)' % (epoch, np.mean(aerrs), len(aerrs)))
            
            #if not model.debug:
            #    theta_i = sess.run(model.theta)
            #    pathlib.Path('%s_train/%s' % (dataset, source)).mkdir(exist_ok=True, parents=True)
            #    sio.savemat('%s_train/%s/%d.mat' % (dataset, source, epoch), mdict={'theta': theta_i})
           
if __name__ == '__main__':
    main()
