import numpy as np
import os, sys

best = np.inf
best_dim = None
best_bs = None

files = [f for f in os.listdir('.') if '_search' in f]

def get_f(filenames):
    f = open(filenames,'r')
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    try:
        loss = float(lines[1].split(':')[1])
    except:
        loss = np.inf
    bs = lines[2]
    dim = lines[3]
    return loss, bs, dim

for f in files:
    loss, bs, dim = get_f(f)
    if loss < best:
        best = loss
        best_bs = bs
        best_dim = dim

print('best loss: ',loss,
      '\nbest bs: ', best_bs,
      '\nbest dim: ',best_dim)

