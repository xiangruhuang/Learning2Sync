import sys
with open(sys.argv[1], 'r') as fin:
    lines = fin.readlines()

n = int(sys.argv[2])
lines_per_file = (len(lines) + n - 1) // n
for i in range(n):
    lines_i = lines[lines_per_file*i:lines_per_file*(i+1)]
    with open('%s.%d.sh' % (sys.argv[1], i), 'w') as fout:
        fout.write('#!/bin/bash\n')
        fout.write('export PATH=${PATH}:/usr/local/bin\n')
        fout.write('source /u/xrhuang/anaconda3/etc/profile.d/conda.sh\n')
        fout.write('export PATH=/u/xrhuang/anaconda3/bin:${PATH}\n')
        fout.write('export PKG_CONFIG_PATH=/u/xrhuang/anaconda3/envs/py36/lib/pkgconfig/\n')
        fout.write('source activate py36\n')
        fout.writelines(lines_i)
