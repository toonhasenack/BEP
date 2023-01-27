from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
from Functions.Hamiltonian_Solver import *
from tqdm import tqdm


@jit(nopython = True)
def d(g1, g2):
    return np.sum(np.abs(subtract(g1,g2)))

@jit(nopython = True)
def GradientDescent(n,dim,t,z,U_func,V_func,mu,it,g_exp,times = 100):
    L,M = dim
    it = 10
    dpmax = 1e-2
    dpmin = 1e-10
    alpha = 1e-3
    p = zeros(2) + np.array([L/10, L/10])
    dp = zeros(2)+1e-2
        
    _, g, _, _ = solve(n,dim,t,z,U_func,V_func,p,mu,it)
    g -= 1/2
    prev = d(g, g_exp)
    errors = np.zeros(times-2)
    for i in range(1,times-1):
        for j in range(2):
            p[j] += dp[j]
            _, g, _, _ = solve(n,dim,t,z,U_func,V_func,p,mu,it)
            g -= 1/2
            now = d(g, g_exp)
            
            if (now == prev) | (dp[j] < dpmin) :
                dp[j] = dpmin
            elif abs(dp[j]) >= dpmax:
                grad = (now-prev)/dp[j]
                grad = grad/np.abs(grad)
                dp[j] = -grad*dpmax
            else:
                grad = (now-prev)/dp[j]
                dp[j] = -alpha*grad
            prev = now
        errors[i-1] = prev
    return p, errors