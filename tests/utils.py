import numpy as np

def generate_checkerboard(size, d=3):
    if d==2:
        cb = np.zeros([size, size])
        a, b = np.meshgrid(range(size), range(size), indexing='ij')
        cb[(a + b) % 2 == 0] = 1
    if d==3:
        cb = np.zeros([size, size, size])
        a, b, c = np.meshgrid(range(size), range(size), range(size), indexing='ij')
        cb[(a + b + c) % 2 == 0] = 1
    return cb