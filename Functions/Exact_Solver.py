from numpy import *
import matplotlib.pyplot as plt

def init_full(n,L,t,U,V,mu):
    I = identity(n)
    a = zeros((n,n))
    c = zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j == i + 1:
                a[i,j] = sqrt(i+1)
            if j == i - 1:
                c[i,j] = sqrt(i)
    A = []
    C = []
    N = []
    for i in range(L):
        s = ["I"]*L
        s[i] = "a" 

        string = s[0]
        for j in range(1,L):
            string += "," + s[j] + ")"

        exec("A.append(" + (L-1)*"kron(" + string + ")")
        C.append(A[-1].T)
        N.append(matmul(C[-1],A[-1]))
    
    H = zeros((n**L,n**L))
    for i in range(L):
        l = 1 + i
        for j in range(L):
            if (abs(j-i) == 1) or (abs(j-i) == L-1):
                H -= t*matmul(C[i], A[j]) 
            if j == i:
                H += 1/2*U(l)*multiply(N[i], N[i] - 1) + V(l)*N[i] - mu*N[i]
                
    return H

def diagonalize(n,L,t,U,V,mu):
    H = init_full(n,L,t,U,V,mu)
    
    E, psi = linalg.eig(H)
    
    k = where(E == min(E))[0][0]
    psi_k = psi[:,k].reshape([n]*L)
    
    uncond_prob = zeros((L,n))
    for i in range(L):
        string = list(L*":,")
        string[2*i] = 'j'
        string = ''.join(string)
        for j in range(n):
            exec(f"uncond_prob[i,j] = sum(psi_k[{string}]**2)")
    
    return H, E, psi, k, uncond_prob

def plot_result(prob, n, L):
    fig = plt.figure(figsize = (9,6))
    ax = plt.axes()
    plt.imshow(prob.T,extent=[0,L,n,0])#, interpolation = 'bicubic')
    ax.invert_yaxis()
    plt.colorbar()
    plt.xticks(linspace(1, L, L));
    plt.yticks(linspace(0, n, n+1));
    ax.set_xlabel(r"Site $\ell$", size = 20)
    ax.set_ylabel(r"Occupancy $|g_\ell\rangle$", size = 20)
    return fig,ax