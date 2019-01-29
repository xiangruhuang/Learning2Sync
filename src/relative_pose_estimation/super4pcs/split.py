import sys
with open(sys.argv[1], 'r') as fin:
    lines = fin.readlines()

n = int(sys.argv[2])
#lines_per_file = (len(lines) + n - 1)// n
for i in range(n):
    #lines_i = lines[lines_per_file*i:lines_per_file*(i+1)]
    lines_i = lines[i::n]
    with open('%s.%d.sh' % (sys.argv[1], i), 'w') as fout:
        fout.write('#!/bin/bash\n')
        fout.writelines(lines_i)
