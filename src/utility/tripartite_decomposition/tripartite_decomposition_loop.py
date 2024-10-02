import numpy as np
from scipy.linalg import svd
import time 
from sklearn.utils.extmath import randomized_svd
from .. import debug_levels

def get_psi(A,S,B):
    psi_p = np.tensordot(S,B,axes=[2,1]) # i a j b
    psi_p = np.tensordot(A,psi_p,axes=[(1,2),(0,2)]) # i a j b
    return(psi_p/np.linalg.norm(psi_p))

def rq(A):
    Q,R = np.linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    return R[:,::-1],Q[::-1,:]

def svd_truncated(A, chi_max,p_trunc): 
    #X,Y,Z = svd(A,full_matrices=False)
    X,Y,Z = randomized_svd(A,n_components=chi_max, random_state=0)
    chi_new = np.min([np.sum(Y/np.linalg.norm(Y)>p_trunc), chi_max])
    Y = Y[:chi_new] / np.linalg.norm(Y[:chi_new])
    
    return X[:,:chi_new], np.diag(Y), Z[:chi_new,:]
    
def split_psi(psi, dL, dR, truncation_par={'chi_max':10, 'p_trunc':1e-8,}, max_iter=100, eps = 1e-10, init ='psi0', log_iterates=False): 
    
    #     S     l3   B 
    #   j ---\------/--- mR k
    #      l2 \    / l1  
    #          \  /
    #           | A
    #           |
    #            d
    #            i

    chi_max = truncation_par['chi_max']
    d,mL,mR = psi.shape
    
    if dL*dR > d:
        raise ValueError

    if  init == 'random':
        A = 0.5 - np.random.rand(d,dL,dR)
        S = 0.5 - np.random.rand(dL,mL,chi_max)
        B = 0.5 - np.random.rand(dR,chi_max,mR)
    else:
        if d < mL*mR:
            rho = np.dot(psi.reshape((d, mL*mR)), psi.reshape((d, mL*mR)).T)
            p, X = np.linalg.eigh(rho)
            perm = np.argsort(-p)
            X = X[:, perm]
            #p = p[perm]
        else:
            X, Y, Z = svd(psi.reshape(d,mL*mR), full_matrices=True)
            #p = Y**2
        
        #opt = np.sqrt(np.sum(p[:dL*dR]))
        A = X[:, :dL*dR].reshape(-1,dL,dR)
    
        theta = np.tensordot(A.conj(),psi,axes=(0,0)).reshape(dL,dR,mL,mR)
        theta = theta.transpose(0,2,1,3).reshape(dL*mL,dR*mR)    
    
        X,Y,Z = svd_truncated(np.ascontiguousarray(theta),chi_max,truncation_par['p_trunc'])
        
        S = (X@Y).reshape(dL,mL,-1)
        B = Z.reshape(-1,dR,mR).transpose(1,0,2)

    info = {}
    if log_iterates:
        info["As"] = []
        info["Bs"] = []
        info["Ss"] = []
    
    m = 0
    go = True
    S2s = []
    while m < max_iter and go:
        theta = np.tensordot(psi,S,axes = [1,1])                        # d mR dL a
        theta = np.tensordot(theta,B,axes = [(1,3),(2,1)])              # d dL dR
        theta = theta.reshape(d,dL*dR)                                  # d dL*dR
    
        X, Y, Z = svd(np.ascontiguousarray(theta),full_matrices=0)
        A = (X@Z).reshape(d,dL,dR)
    
        theta = np.tensordot(psi, A, axes=[0,0])                        # mL mR dL dR
        theta = theta.transpose(2,0,3,1)                                # dL mL dR mR
        theta = theta.reshape(dL*mL,dR*mR)

        X,Y,Z = svd_truncated(np.ascontiguousarray(theta),chi_max,truncation_par['p_trunc'])
        
        S = (X@Y).reshape(dL,mL,-1)
        B = Z.reshape(-1,dR,mR).transpose(1,0,2)

        S2s.append(np.sum(Y)) 
        m+=1
        if m > 1:
            go = np.abs(S2s[-1] - S2s[-2]) >= eps

        if log_iterates:
            info["As"].append(A)
            info["Bs"].append(B)
            info["Ss"].append(S)
    
    info['s_Lambda'] = np.diag(Y)
    return A, S, B, info

def tripartite_decomposition(T, D1, D2, chi, N_iters=100, eps=1e-9, p_trunc=1e-8, initialize="psi0", debug_dict=None):
    chi_1, chi_2, chi_3 = T.shape
    T = T.transpose(0, 2, 1) # chi_1 chi_2 chi_3 = d mR mL -> d mL mR
    # D1 D2 = dR dL
    # chi = chi_max
    log_iterates = debug_levels.check_debug_level(debug_dict, debug_levels.DebugLevel.LOG_PER_ITERATION_DEBUG_INFO_DISENTANGLER_TRIPARTITE_DECOMPOSITION)
    A, S, B, info = split_psi(psi=T, dL=D2, dR=D1, truncation_par={'chi_max':chi, 'p_trunc':p_trunc,}, max_iter=N_iters, eps=eps, init=initialize, log_iterates=log_iterates)
    A = A.transpose(0, 2, 1) # A: d dL dR = chi_1 D2 D1 -> chi_1 D1 D2
    B = B.transpose(0, 2, 1) # B: dR chi mR = D1 chi chi_2 -> D1 chi_2 chi
    C = S.transpose(0, 2, 1) # S: dL mL chi = D2 chi_3 chi -> D2 chi chi_3
    if log_iterates:
        debug_dict["tripartite_decomposition_iterates_A"] = []
        debug_dict["tripartite_decomposition_iterates_B"] = []
        debug_dict["tripartite_decomposition_iterates_C"] = []
        for i in range(len(info["As"])):
            debug_dict["tripartite_decomposition_iterates_A"].append(info["As"][i].transpose(0, 2, 1)) # A: d dL dR = chi_1 D2 D1 -> chi_1 D1 D2
            debug_dict["tripartite_decomposition_iterates_B"].append(info["Bs"][i].transpose(0, 2, 1)) # B: dR chi mR = D1 chi chi_2 -> D1 chi_2 chi
            debug_dict["tripartite_decomposition_iterates_C"].append(info["Ss"][i].transpose(0, 2, 1)) # S: dL mL chi = D2 chi_3 chi -> D2 chi chi_3
    return A, B, C



