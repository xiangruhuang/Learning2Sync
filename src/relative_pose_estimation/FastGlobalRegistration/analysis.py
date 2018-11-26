import numpy as np

with open('haha', 'r') as fin:
    lines = fin.readlines()
    aerrs = []
    terrs = []
    for line in lines:
        sigma, aerr, terr = [float(token.split('=')[-1].strip()) for token in line.strip().split(',')]
        
        print(line, sigma, aerr, terr)
        if sigma < 0.5:
            aerrs.append(aerr)
            terrs.append(terr)
    print(np.mean(aerrs), np.mean(terrs))
