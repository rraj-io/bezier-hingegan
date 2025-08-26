'''
Rational Bezier Curves
'''

import numpy as np
from scipy.special import binom

def bernstein(t, i, n):
    return binom(n, i) * t**i * (1-t)**(n-i)

def rational_bezier_curve(t, p, w):
    #print(f"Shape of p is {p.shape}")
    #print(f"Shape of w is {w.shape}")
    p = np.asarray(p)
    w = np.asarray(w).flatten()
    #n = p.shape[0] -1
    
    #w = w.reshape(32, -1)
    #print(f"Shape of p is {p.shape}")
    #print(f"Shape of w is {w.shape}")

    #assert len(p) == len(w)
    n = p.shape[0] - 1
    numerator = np.zeros(2)
    denominator = 0

    for i in range(n+1):
        numerator += bernstein(t, i, n) * p[i] * w[i]
        denominator += bernstein(t, i, n) * w[i]

    return numerator / denominator