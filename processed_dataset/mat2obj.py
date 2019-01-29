import sys
import scipy.io as sio
import glob

## OBJ file
#v -0.3925 -0.8111 2.0260

s = int(sys.argv[1])

with open('scenes', 'r') as fin:
  scene_id = fin.readlines()[s].strip()

mats = glob.glob('scannet/%s/*.mat' % scene_id)

for mat_f in mats:
  obj = mat_f.replace('.mat', '.obj')
  mat = sio.loadmat(mat_f)
  #print(mat.keys())
  with open(obj, 'w') as fout:
    fout.write('# OBJ file\n')
    v = mat['vertex']
    assert v.shape[0] == 3
    for i in range(v.shape[1]):
      fout.write('v %.4f %.4f %.4f\n' % (v[0, i], v[1, i], v[2, i]))
    
