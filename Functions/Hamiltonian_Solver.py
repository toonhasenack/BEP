from numpy import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import jit, njit
import time 
import cProfile
import io
import pstats

@jit(nopython = True)
def expm(a):
    n = a.shape[0]
    q = 6
    a2 = a.copy ( )
    a_norm = np.linalg.norm ( a2, np.inf )
    ee = ( int ) ( np.log2 ( a_norm ) ) + 1
    s = max ( 0, ee + 1 )
    a2 = a2 / ( 2.0 ** s )
    x = a2.copy ( )
    c = 0.5
    e = np.eye ( n, dtype = np.complex64 ) + c * a2
    d = np.eye ( n, dtype = np.complex64 ) - c * a2
    p = True

    for k in range ( 2, q + 1 ):
        c = c * float ( q - k + 1 ) / float ( k * ( 2 * q - k + 1 ) )
        x = np.dot ( a2, x )
        e = e + c * x
        
        if ( p ):
            d = d + c * x
        else:
            d = d - c * x

        p = not p
    #  E -> inverse(D) * E
    e = np.linalg.solve ( d, e )
    #  E -> E^(2*S)
    for k in range ( 0, s ):
        e = np.dot ( e, e )
    return e
    
@jit(nopython = True)
def init_i(n):
    I = identity(n)
    a_hat = zeros((n,n))
    c_hat = zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j == i + 1:
                a_hat[i,j] = sqrt(i+1)
            if j == i - 1:
                c_hat[i,j] = sqrt(i)
    
    n_hat = c_hat @ a_hat
    return a_hat, c_hat, n_hat

@jit(nopython = True)
def Hamiltonian(t,z,U,V,mu,param,a_hat,c_hat,n_hat):
    return 1/2*U*multiply(n_hat, n_hat - 1) + V*n_hat - mu*n_hat - t*z*param*(param + a_hat + c_hat)

@jit(nopython = True)
def Perturbation(V_t,n_hat):
    return V_t*n_hat

@jit(nopython = True)
def solve_i(H_i):
    E_i, phi_i = linalg.eig(H_i)
    k = where(E_i == min(E_i))[0][0]
    phi_k = phi_i[:,k].reshape(H_i.shape[0],)
    E_k = E_i[k]
    
    return phi_k, E_k

@jit(nopython = True)
def solve(n,dim,t,z,U,V,V_p,mu,it=10):   
    L,M = dim
    a_hat, c_hat, n_hat = init_i(n)
    param = zeros((L,M)) + 1
    psi = np.zeros((L,M,n))
    g = np.zeros((L,M))
    E = zeros(L*M)
    Es = zeros(it)
    for i in range(it):
        for j in range(L):
            for k in range(M):
                x = np.array([[j+1, k+1],])
                H_i = Hamiltonian(t,z,U(x),V(x, V_p),mu,param[j,k],a_hat,c_hat,n_hat).astype("float64")
                phi, E[j] = solve_i(H_i)
                param[j,k] = phi @ a_hat @ phi
        Es[i] = np.sum(E)
    
    for i in range(L):
        for j in range(M):
            x = np.array([[i+1, j+1],])
            H_i = Hamiltonian(t,z,U(x),V(x, V_p),mu,param[i,j],a_hat,c_hat,n_hat)
            phi, _ = solve_i(H_i)
            g[i,j] = phi @ n_hat @ phi
            psi[i,j] = phi
        
    return psi, g, param, Es

@jit(nopython = True)
def dynamics(n,dim,t,z,U,V,V_p,V_t,mu,psi,param,t_interval=[0,1],t_points=1000):
    L,M = dim
    a_hat, c_hat, n_hat = init_i(n)
    #first we generate the Hamiltonian of each lattice site
    H = np.zeros((L,M,n,n))
    for i in range(L):
        for j in range(M):
            x = np.array([[i+1, j+1],])
            H[i,j] = Hamiltonian(t,z,U(x),V(x, V_p),mu,param[i,j],a_hat,c_hat,n_hat)
    
    t_s = linspace(t_interval[0], t_interval[1], t_points)
    dt = (t_interval[1] - t_interval[0])/t_points
    psi_s = np.zeros((t_points,L,M,n), dtype = 'complex_')
    psi_I_s = np.zeros((t_points,L,M,n), dtype = 'complex_') 
    g_s = np.zeros((t_points,L,M))
    for i in range(t_points):
        for l in range(L):
            for m in range(M):
                U_hat = expm(-1j*t_s[i]*H[l,m])
                if i == 0:
                    psi_I_s[i,l,m] = psi[l,m].astype('complex_')
                else:
                    x = np.array([[l + 1, m + 1],])
                    H_t = Perturbation(V_t(x, t_s[i]), n_hat)
                    H_t_I = np.conj(U_hat) @ H_t.astype('complex_') @ U_hat
                    psi_I_s[i,l,m] = psi_I_s[i-1,l,m] - 1j*dt*H_t_I @ psi[l,m].astype('complex_')

    for i in range(t_points):
        for l in range(L):
            for m in range(M):
                U_hat = expm(-1j*t_s[i]*H[l,m])
                psi_s[i,l,m] = U_hat @ psi_I_s[i,l,m]
                g_s[i,l,m] = np.real(np.conj(psi_s[i,l,m]) @ n_hat.astype('complex_') @ psi_s[i,l,m])     
    return psi_s, g_s