import numpy as np
from .. import utility
from .. import debug_levels

def disentangle(theta, eps=1e-10, N_iters=200, min_iters=0, debug_dict=None):
    """
    Disentangles the given wavefunction theta by minimizing the renyi-2-entropy, using the fast power iteration method. This disentangler was introduced in
    [1] J. Hauschild, E. Leviatan, J. H. Bardarson, E. Altman, M. P. Zaletel, and F. Pollmann: "Finding purifications with minimal entanglement", https://arxiv.org/abs/1711.01288 
    
    Parameters
    ----------
    theta : np.ndarray of shape (l, i, j, r)
        Wavefunction tensor to be disentangled.
    eps : float, optional
        After the difference in renyi entropy of two consecutive iterations is smaller than this threshhold value,
        the algorithm terminates. Default: 1e-15.
    N_iters : int, optional
        Maximum number of iterations the algorith is run for. Default: 200.
    min_iters : int, optional
        Minimum number of iterations, before the eps condition leads to a termination of the algorithm.
        Mostly for debugging purposes. Default: 0.
    debug_dict : dictionary, optional
        dictionary in which debug information is saved. Default: None.

    Returns
    -------
    U_final : np.ndarray of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    # Helper function
    def _U2(theta):
        """
        Helper function used in the disentangling algorithm.

        Parameters
        ----------
        theta : np.ndarray of shape (l, d1, d2, r)
            current wavefunction tensor
        
        Returns
        -------
        s : float
            renyi 2 entropy of the wavefunction
        u : np.ndarray of shape (d1*d2, d1*d2)
            update for disentangling unitary
        """
        chi = theta.shape
        rhoL = np.tensordot(theta, np.conj(theta), axes = [[2, 3], [2, 3]]) # ml d1 [d2] [mr]; ml* d1* [d2*] [mr*] -> ml d1 ml* d1* { D^9 }

        dS = np.tensordot(rhoL, theta, axes = [[2, 3], [0, 1] ]) # ml d1 [ml*] [d1*]; [ml] [d1] d2 mr -> ml d1 d2 mr { D^9 }
        dS = np.tensordot( np.conj(theta), dS, axes = [[0, 3], [0, 3]]) # [ml] d1 d2 [mr]; [ml*] d1* d2* [mr*] { D^8 }

        dS = dS.reshape((chi[1]*chi[2], -1))
        s2 = np.trace( dS )
        
        X, Y, Z = utility.safe_svd(dS)
        return -np.log(s2), (np.dot(X, Z).T).conj()
    # Initialize
    _, d1, d2, _ = theta.shape
    U = np.eye(d1*d2, dtype = theta.dtype) # { D^4 }
    # debug info
    log_debug_info = debug_levels.check_debug_level(debug_dict, debug_levels.DebugLevel.LOG_PER_ITERATION_DEBUG_INFO_DISENTANGLER_TRIPARTITE_DECOMPOSITION)
    if log_debug_info:
        debug_dict["disentangler_iterates"] = []
    # Main loop
    m = 0
    go = True
    Ss = []
    while m < N_iters and (go or m < min_iters):
        s, u = _U2(theta) 
        U = np.dot(u, U)
        if log_debug_info:
            debug_dict["disentangler_iterates"].append(U.reshape((d1,d2,d1,d2)))
        u = u.reshape((d1,d2,d1,d2))
        theta = np.tensordot(u.conj(), theta, axes = [[2, 3], [1, 2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 1:
            go = Ss[-2] - Ss[-1] > eps 
        m+=1
    # Save debug information
    if debug_levels.check_debug_level(debug_dict, debug_levels.DebugLevel.LOG_DISENTANGLER_TRIPARTITE_DECOMPOSITION_ITERATION_INFO):
        utility.append_to_dict_list(debug_dict, "N_iters_disentangler", m)
    # Return result
    return np.reshape(np.conj(U), (d1, d2, d1, d2))