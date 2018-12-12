import numpy as np
from scipy.sparse import csc_matrix
from algebra import transform
import cv2
import os

class Mesh:
    def __init__(self, vertex=None, faces=None):
        self.vertex = vertex
        self.faces = faces
        self.vertexNormal = None

    def writeOBJ(self, filename):
        nverts = self.vertex.shape[1]
        with open(filename, 'w') as f:
            f.write("# OBJ file\n")
            for v in range(nverts):
                f.write("v %.4f %.4f %.4f\n" % (self.vertex[0,v],self.vertex[1,v],self.vertex[2,v]))

    def readOBJ(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            assert(lines[0] == "# OBJ file\n")
            lines = lines[1:]
            n = len(lines)
            pcs = []
            for v in range(n):
                tmp = lines[v].strip().split(" ")[1:]
                pcs.append(np.array([float(tmp[0]),float(tmp[1]),float(tmp[2])]).reshape(1,3))
        pcs = np.concatenate(pcs).T
        self.vertex = pcs

    def getKeypoint(self,rs):
        grays= cv2.cvtColor(rs,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, dess) = sift.detectAndCompute(grays, None)
        if not len(kps):
            raise Exception("no keypoint found")
        pts=np.zeros([len(kps),2])
        for j,m in enumerate(kps):
            pts[j,:] = m.pt
        
        pts=pts.astype('int')
        return pts,dess

    def readDepth(self, filename, intrinsic):
        '''
        # image: [h, w], assume invalid pixel has value '0'
        # intrisic: [3, 3]
        # ret: Mesh
        '''
        image = cv2.imread(filename,2) / 1000.
        #imageuint8 = ((image-image.min())/(image.max()-image.min())*255).astype('uint8')
        #imageuint8 = np.tile(imageuint8[:,:,np.newaxis],[1,1,3])
        #pts,dess = self.getKeypoint(imageuint8)
        #self.ptsIdx = pts[:,1]*image.shape[1]+pts[:,0]
        #feat = np.zeros([image.shape[0]*image.shape[1], 128])
        #feat[self.ptsIdx] = dess
        #self.siftFeat = feat 

        self.height, self.width = image.shape
        h, w = image.shape
        ys, xs = np.meshgrid(range(h), range(w), indexing='ij')
        idx = np.array(range(h * w)).reshape(h, w)
        vertex = np.zeros([h * w, 3])
        
        vertex[:,2] = image.flatten()
        vertex[:,0] = ((xs - intrinsic[0,2]) / intrinsic[0,0]).flatten() * vertex[:,2]
        vertex[:,1] = ((ys -intrinsic[1,2]) / intrinsic[1,1]).flatten() * vertex[:,2]
        
        # Labeling each pixel, invalid pixels are labeled '0'
        label = (np.power(vertex[:,2],2) > 0)
        validId = (label > 0)
        label[validId] = np.array(range(validId.sum()))
        
        id1 = np.array(range(0,w-1))
        id2 = np.array(range(1,w))
        id3 = np.array(range(0,h-1))
        id4 = np.array(range(1,h))
        idtl = idx[id3[:,np.newaxis], id1].reshape(1,-1)
        idtr = idx[id3[:,np.newaxis], id2].reshape(1,-1)
        idbl = idx[id4[:,np.newaxis], id1].reshape(1,-1)
        idbr = idx[id4[:,np.newaxis], id2].reshape(1,-1)
        
        faces = np.zeros([3,idtl.shape[1]*2])
        faces[:,:idtl.shape[1]] = np.concatenate((idtl,idbl,idbr),0)
        faces[:,idtl.shape[1]:] = np.concatenate((idtl,idbr,idtr),0)
        faces = faces.astype('int')
        
        ## Delete faces with invalid vertex
        faces = faces[:,label[faces].min(0) > 0]
        
        ## Delete huge faces
        e12 = np.linalg.norm(vertex[faces[1,:],:] - vertex[faces[0,:],:],axis=1)
        e23 = np.linalg.norm(vertex[faces[2,:],:] - vertex[faces[1,:],:],axis=1)
        e31 = np.linalg.norm(vertex[faces[0,:],:] - vertex[faces[2,:],:],axis=1)
        med = np.median(np.concatenate((e12,e23,e31)))
        validId = np.logical_and((e12 <= med), (e23 <= med), (e31 <= med))
        faces = faces[:, validId]
        
        self.vertex=vertex.T
        self.faces=faces
        
        # Clean unreferenced vertex
        self.clean()
        
        # Compute Normal
        self.computeNormal()

    @classmethod
    def read(cls, filename, mode='obj',intrinsic=None):
        mesh = cls()
        if mode == 'obj':
            mesh.readOBJ(filename)
        elif mode == 'depth':
            mesh.readDepth(filename,intrinsic)
        return mesh

    def write(self, filename):
        _, file_extension = os.path.splitext(filename)
        if file_extension == '.obj':
            self.writeOBJ(filename)
        else:
            print('not implemented')

    def transform(self, R):
        """
        # R: [4,4]
        """
        self.vertex = transform(R, self.vertex)

    def clean(self):
        numV = self.vertex.shape[1]
        label = np.zeros([numV])
        label[self.faces[0,:]] = 1
        label[self.faces[1,:]] = 1
        label[self.faces[2,:]] = 1
        validId = (label > 0)
        validIdx = []
        for i, xi in enumerate(label):
            if xi > 0:
                validIdx.append(i)
        self.validIdx = np.array(validIdx)
        label[validId] = np.array(range(validId.sum()))
        self.vertex = self.vertex[:,validId]
        self.faces = label[self.faces].astype('int')
        #self.ptsIdx = label[self.ptsIdx].astype('int')
        #self.siftFeat=self.siftFeat[validId]

    def computeNormal(self):
        e12 = self.vertex[:,self.faces[1,:]] - self.vertex[:,self.faces[0,:]]
        e13 = self.vertex[:,self.faces[2,:]] - self.vertex[:,self.faces[0,:]]
        fNormal = np.cross(e12, e13, axis=0)
        
        numV = self.vertex.shape[1]
        numF = self.faces.shape[1]
        rows = np.kron(np.array(range(numF)),np.ones([1,3])).flatten()
        cols = self.faces.T.flatten()
        vals = np.ones([3*self.faces.shape[1]])
        AdjMatrix = csc_matrix((vals,(rows,cols)),(numF, numV))
        fNormal = (csc_matrix(fNormal)*AdjMatrix).todense()

        fNormal = fNormal / np.linalg.norm(fNormal,axis=0)
        self.vertexNormal = fNormal
