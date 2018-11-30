import numpy as np
from scipy.stats import special_ortho_group as so
from scipy.linalg import sqrtm
import glob
from scipy.linalg import block_diag
import sys
sys.path.append('../../../')
from util import angular_distance_np
import scipy.io as sio
import pathlib

def __decompose__(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

def __pack__(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T

def inverse(T):
    R, t = __decompose__(T)
    invT = np.zeros((4, 4))
    invT[:3, :3] = R.T
    invT[:3, 3] = -R.T.dot(t)
    invT[3, 3] = 1
    return invT

"""
    Find a matrix Q \in O(n) such that \|A Q - B\|_F is minimized
    equivalent to maximize trace of (Q^T A^T B)
"""
def project(A, B):
    X = A.T.dot(B)
    U, S, VT = np.linalg.svd(X)
    Q = U.dot(VT)
    return Q

"""
    Find a matrix Q \in SO(n) such that \|Q - X\|_F is minimized
    equivalent to project(I, X)
"""
def project_so(X):
    d = X.shape[0]
    assert X.shape[1] == d
    Q = project(np.eye(d), X)
    Q = Q * np.linalg.det(Q)
    return Q

def generate_synthetic(n, sigma):
    T = np.zeros((n, 4, 4))
    X = so.rvs(dim=3, size=n)
    T[:, :3, :3] = X
    # u, sigma, v = np.linalg.svd(T[0])
    T[:, :3, 3] = np.random.randn(n, 3)
    T[0, :3, 3] = 0.0
    T[:, 3, 3] = 1
    edges = []
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            Tij = T[j].dot(inverse(T[i]))
            Rij, tij = __decompose__(Tij)
            Rij = Rij + np.random.randn(3, 3) * sigma
            Rij = project_so(Rij)
            tij = tij + np.random.randn(3) * sigma
            Tij = __pack__(Rij, tij)
            edge = {'src':i,
                    'tgt':j,
                    'R': Rij,
                    't': tij,
                    'weight': 1.0}
            edges.append(edge)
    edges = np.array(edges)
    return n, edges, T

"""
    Construct Normalized Adjacency Matrix 
    Anorm(i, j) = {
        wij Rij.T / sqrt(di dj),  if (i, j) is an edge
        0, o.w. 
    }
"""
def __normalized_adjacency__(n, edges):
    A = np.zeros((3*n, 3*n))
    deg = np.zeros(n)
    Adj = np.zeros((n, n))
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['weight']
        A[(i*3):(i+1)*3, (j*3):(j+1)*3] = weight*Rij.T
        A[(j*3):(j+1)*3, (i*3):(i+1)*3] = weight*Rij
        deg[i] += weight
        deg[j] += weight

    Dinv = np.kron(np.diag(1.0/deg), np.eye(3))

    Anorm = sqrtm(Dinv).dot(A).dot(sqrtm(Dinv))
    return Anorm, deg

"""
    Construct Normalized Adjacency Matrix 
    Anorm(i, j) = {
        wij Rij.T / sqrt(di dj),  if (i, j) is an edge
        0, o.w. 
    }
"""
def laplacian(n, edges):
    A = np.zeros((3*n, 3*n))
    deg = np.zeros(n)
    Adj = np.zeros((n, n))
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['weight']
        A[(i*3):(i+1)*3, (j*3):(j+1)*3] = -weight*Rij.T
        A[(j*3):(j+1)*3, (i*3):(i+1)*3] = -weight*Rij
        deg[i] += weight
        deg[j] += weight

    for i in range(n):
        A[i*3:(i+1)*3, i*3:(i+1)*3] = np.eye(3) * deg[i]

    #Dinv = np.kron(np.diag(1.0/deg), np.eye(3))

    #Anorm = sqrtm(Dinv).dot(A).dot(sqrtm(Dinv))
    return A, deg

def from_laplacian(L):
    n = L.shape[0]//3
    #deg = np.diag(L)[::3]
    #Dinv = np.kron(np.diag(1.0/deg), np.eye(3))
    
    lamb, V = np.linalg.eigh(L)

    #print(lamb)
    #lamb = abs(lamb)
    #idx = np.argsort(lamb)
    #print(lamb[idx])
    eigengap = lamb[-3] - lamb[-4]
    for i in range(n):
        if V[0, i] < 0.0:
            V[:, i] = -V[:, i]
    if np.linalg.det(V[:3, :3]) < 0:
        V[:, :] = -V[:, :]
    #dsqrt = np.sqrt(deg.sum())
    #V = sqrtm(Dinv).dot(V) * dsqrt
    
    A = []
    R = []
    for i in range(n):
        Ai = V[i*3:(i+1)*3, :3]
        A.append(Ai)
        Ri = project_so(Ai)
        R.append(Ri)

    R = np.array(R)

    return R, eigengap, A, lamb, V

""" Estimate Absolute Rotation from Relative Rotation
    n: number of vertices
    edges: array of np.object, each contains items [i, j, Tij, weight]
"""
def Spectral(n, edges):
    L, deg = laplacian(n, edges)
    #Lnorm = np.eye(n*3)-Anorm
    R, eigengap, A, lamb, V = from_laplacian(L)
    return R, eigengap
    #Dinv = np.kron(np.diag(1.0/deg), np.eye(3))

    #lamb, V = np.linalg.eigh(L)
    ##print(lamb)
    ##lamb = abs(lamb)
    ##idx = np.argsort(lamb)
    ##print(lamb[idx])
    #eigengap = lamb[-3] - lamb[-4]
    #V = V[:, :3]
    #dsqrt = np.sqrt(deg.sum())
    ##V = sqrtm(Dinv).dot(V) * dsqrt

    #R = []
    #for i in range(n):
    #    Ri = V[i*3:(i+1)*3, :]
    #    Ri = project_so(Ri)
    #    R.append(Ri)

    #R = np.array(R)

    #return R, eigengap

#"""
#    Solve min_{ti, tj} \sum_{i, j} wij \| Rj tij + tj - ti \|^2
#    Gradient Descent (only for debug)
#"""
#def __LeastSquaresGD__(n, edges, R):
#    t = np.zeros((n, 3))
#    lr = 1e-5
#    for itr in range(10000):
#        grad = np.zeros((n, 3))
#        loss = 0.0
#        for edge in edges:
#            i = edge['i']
#            j = edge['j']
#            Rij = edge['Tij'][:3, :3]
#            tij = edge['Tij'][:3, 3]
#            weight = edge['weight']
#            loss += weight * np.linalg.norm(Rij.dot(t[i]) + tij - t[j], 2) ** 2
#            grad[i, :] += 2.0*weight*(t[i] + Rij.T.dot(tij) - Rij.T.dot(t[j]))
#            grad[j, :] += 2.0*weight*(t[j] - Rij.T.dot(t[i]) - tij)
#        t = t - grad * lr
#        if itr % 100 == 0:
#            print('iter=%d, loss=%f' % (itr, loss))
#    return t

"""
    Solve min_{ti, tj} \sum_{i, j} wij \| Rj tij + tj - ti \|^2
    By solving linear equation At = b.
"""
def LeastSquares(n, edges):
    Anorm, deg = __normalized_adjacency__(n, edges)
    Lnorm = np.eye(n*3) - Anorm
    #_, nullL = np.eigh(Anorm)
    #nullL = nullL(:, -3:)
    D = np.kron(np.diag(deg), np.eye(3))
    L = sqrtm(D).dot(Lnorm).dot(sqrtm(D))
    #Lnorm = np.eye(3)
    #Lnorm = np.eye(3)

    b = np.zeros(n*3)
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        tij = edge['t']
        weight = edge['weight']
        
        #L[i*3:(i+1)*3, j*3:(j+1)*3] -= weight * Rij.T
        #L[j*3:(j+1)*3, i*3:(i+1)*3] -= weight * Rij
        #L[i*3:(i+1)*3, i*3:(i+1)*3] += weight * np.eye(3)
        #L[j*3:(j+1)*3, j*3:(j+1)*3] += weight * np.eye(3)
        b[i*3:(i+1)*3] += weight*(-Rij.T).dot(tij)
        b[j*3:(j+1)*3] += weight*tij

    t = np.linalg.lstsq(L, b)[0]
    #print(L.shape, t.shape)
    #print('Loss=%f' % np.linalg.norm(L.dot(t) - b, 2))
    return t

#def find(x, f):
#    if f[x] == f[f[x]]:
        

#def connected(n, edges):
#    f = np.zeros(n)
#    for i in range(n):
#        f[i] = i
#    for edge in edges:
#        i = edge['src']
#        j = edge['tgt']
#        

def error(T, G):
    aerrs = []
    n = T.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            Ti = T[i, :, :]; Tj = T[j, :, :]
            Gi = G[i, :, :]; Gj = G[j, :, :]
            Rij = Tj.dot(inverse(Ti))[:3, :3]
            Rij_gt = Gj.dot(inverse(Gi))[:3, :3]
            fro = np.linalg.norm(Rij - Rij_gt, 'fro')
            aerr = angular_distance_np(Rij[np.newaxis, :, :], Rij_gt[np.newaxis, :, :]).sum()
            #print(fro, aerr)
            aerrs.append(aerr)
    return np.mean(aerrs)

def coin(p):
    if np.random.uniform() < p:
        return 1
    else:
        return 0

def TransfSync(n, edges):
    R, eigengap = Spectral(n, edges)
    t = LeastSquares(n, edges); t = np.reshape(t, (-1, 3))
    return R, t, eigengap

def max_existing_err(n, edges, R, t):
    max_err = 0.0
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['weight']
        if weight < 1e-10:
            continue
        err = np.linalg.norm(R[j].dot(R[i].T) - Rij, 'fro')
        if err > max_err - 1e-12:
            max_err = err
        
    return max_err 

def truncatedWeightPredict(n, edges, R, t, eps0):
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['weight']
        #if edge['predicted_weight'] < 0.5:
        #    edge['weight'] = 0.
        #    continue
        if weight < 1e-6:
            continue
        err = np.linalg.norm(R[j].dot(R[i].T) - Rij, 'fro')
        if err > eps0 - 1e-12:
            edge['weight'] = 0.0
def reweightEdges(n, edges, R, t, sigma=0.01):
    theta1 = 8
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['predicted_weight']
        if weight < 0.001:
            weight = 0.001
        if weight > 1.0:
            weight = 1.0
        fro2 = np.linalg.norm(R[j].dot(R[i].T) - Rij, 'fro')
        edge['weight'] = (sigma ** theta1) / (sigma ** theta1 + fro2 ** theta1) * (weight ** theta1)
        #print(edge['weight'], edge['predicted_weight'])

def computeStats(n, edges, R, t):
    cover = np.zeros(n, dtype=np.int32)
    numedges = 0
    err_sum = 0.0
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['weight']
        if weight < 1e-10:
            continue
        err = np.linalg.norm(R[j].dot(R[i].T) - Rij, 'fro') ** 2
        err_sum += err
        
        cover[i] += 1
        cover[j] += 1
        numedges += 1
    return min(cover), numedges, err_sum

def dump_edges(n, edges, Tstar, path):
    L = np.zeros((n*3, n*3))
    label = np.zeros((n, n))
    deg = np.zeros(n)
    good = 0
    bad = 0
    for edge in edges:
        src = edge['src']
        tgt = edge['tgt']
        Rij = edge['R']
        Ti = Tstar[src]; Tj = Tstar[tgt]
        Tij_gt = Tj.dot(inverse(Ti))
        aerr = angular_distance_np(Rij[np.newaxis, :, :], Tij_gt[np.newaxis, :3, :3]).sum()
        labelij = edge['weight']
        if labelij > 0.5:
            if aerr < 30.0:
                good += 1
            else:
                bad += 1
        label[src, tgt] = labelij
        label[tgt, src] = labelij
        L[src*3:(src+1)*3, tgt*3:(tgt+1)*3] = -Rij.T
        deg[src] += 1.0
        deg[tgt] += 1.0
    L = L + L.T
    for i in range(n):
        L[i*3:(i+1)*3, i*3:(i+1)*3] = deg[i]*np.eye(3)
    print('good=%d, bad=%d' % (good, bad))
    pathlib.Path('./predict_new/%s' % path).mkdir(exist_ok=True, parents=True)
    sio.savemat('./predict_new/%s.mat' % path, mdict={'Laplacian': L, 'label': label, 'Rstar': Tstar[:, :3, :3]})
    

def fakeTruncatedTransfSync(n, edges, eps0=-1, decay=0.99, Tstar=None, max_iter=10000, cheat=False, pc = None, idx = None, path=None): 
    predictEdgeWeight(n, edges, pc, idx)
    dump_edges(n, edges, Tstar, path)

def edge_quality(n, edges, Tstar, R):
    good_edges = 0.0
    bad_edges = 0.0
    err_bad = 0.0
    err_good = 0.0
    for edge in edges:
        if edge['weight'] < 0.01:
            continue
        sid = edge['src']; tid = edge['tgt']
        Ti = Tstar[sid]; Tj = Tstar[tid]
        Tij_gt = Tj.dot(inverse(Ti))
        Rij_in = edge['R']; tij_in = edge['t']
        Tij_in = __pack__(Rij_in, tij_in)
        aerr_gt = angular_distance_np(Tij_in[np.newaxis, :3, :3], Tij_gt[np.newaxis, :3, :3]).sum()
        Rij = R[tid].dot(R[sid].T)
        aerr = np.linalg.norm(Rij - Rij_in, 'fro') ** 2
        if aerr_gt > 30.0:
            bad_edges += edge['weight']
            err_bad += aerr * edge['weight']
        else:
            good_edges += edge['weight']
            err_good += aerr * edge['weight']
    print('Edge Quality: #good=%f, #bad=%f, mean aerr=(%f, %f)' % (good_edges, bad_edges, err_good / good_edges, err_bad / bad_edges))
    

def TruncatedTransfSync(n, edges, eps0=-1, decay=0.8, Tstar=None, max_iter=10000, cheat=False):
    if cheat:
        for edge in edges:
            #if edge['weight'] < 0.5:
            #    continue
            sid = edge['src']; tid = edge['tgt']
            Ti = Tstar[sid]; Tj = Tstar[tid]
            Tij_gt = Tj.dot(inverse(Ti))
            Rij = edge['R']; tij = edge['t']
            Tij = __pack__(Rij, tij)
            aerr = angular_distance_np(Tij[np.newaxis, :3, :3], Tij_gt[np.newaxis, :3, :3]).sum()
            #p = 0.9
            if aerr > 30.0:
                edge['weight'] = coin(0.03)
            else:
                edge['weight'] = coin(0.8)
            edge['predicted_weight'] = edge['weight']

    """ Edge Quality """
    good_edges = 0.0
    bad_edges = 0.0
    err_bad = 0.0
    err_good = 0.0
    for edge in edges:
        if edge['weight'] < 0.01:
            continue
        sid = edge['src']; tid = edge['tgt']
        Ti = Tstar[sid]; Tj = Tstar[tid]
        Tij_gt = Tj.dot(inverse(Ti))
        Rij = edge['R']; tij = edge['t']
        Tij = __pack__(Rij, tij)
        aerr = angular_distance_np(Tij[np.newaxis, :3, :3], Tij_gt[np.newaxis, :3, :3]).sum()
        if aerr > 30.0:
            #print(sid, tid, edge['predicted_weight'])
            bad_edges += 1
            err_bad += aerr * edge['weight']
        else:
            good_edges += 1
            err_good += aerr * edge['weight']
    print('Edge Quality: #good=%f, #bad=%f' % (good_edges, bad_edges))
    reweight = False
    itr = 0
    while itr < max_iter:
        try:
            R, t, eigengap = TransfSync(n, edges)
        except Exception as e:
            print(e)
            break
        #edge_quality(n, edges, Tstar, R)
        T = np.array([__pack__(R[i], t[i]) for i in range(n)])
        err_gt = -1.0
        if Tstar is not None:
            err_gt = error(T, Tstar)
        
        if eps0 < -0.5:
            eps0 = max_existing_err(n, edges, R, t)

        if reweight:
            reweightEdges(n, edges, R, t, sigma=0.1)
        else:
            truncatedWeightPredict(n, edges, R, t, eps0)

        mindeg, numedges, err_sum = computeStats(n, edges, R, t)
        print('iter=%d, avg(err^2)=%f, eigengap=%f, #edges=%d, min_deg=%f, eps0=%f, aerr_gt=%f' % (itr, err_sum/numedges, eigengap, numedges, mindeg, eps0, err_gt))
    
        """ Skip idle iterations """
        if reweight:
            itr += 1
        else:
            max_err = max_existing_err(n, edges, R, t)
            while (itr < max_iter) and (eps0 > max_err):
                eps0 = eps0 * decay
                itr += 1
        if mindeg == 0:
            break
        if err_sum <= 1e-2:
            break
    return T
        

