import numpy as np
from .. import utility
from ..disentangle import disentangle as disentangle_lib

def tripartite_decomposition(T, D1, D2, chi, N_iters_svd=None, eps_svd=0.0, disentangle=False, disentangle_options={}, debug_dict=None):
    """
    Performs the tripartite decomposition via two consecutive SVDs. Optionally a disentangling step is done before the second SVD.

    Parameters
    ----------
    T : np.ndarray of shape (chi_1, chi_2, chi_3)
        the tensor that is to be decomposed.
    D1 : int
        bond dimension of the leg connecting the A and B tensor.
    D2 : int
        bond dimension of the leg connecting the A and C tensor.
    chi : int
        bond dimension of the leg connecting the B and C tensor.
    N_iters_svd : int or None, optional
        Number of iterations for approximating the SVD, gets passed into split_matrix_iterate_QR().
        if this is set to None, a full SVD is used instead.
    eps_svd : float, optional
        eps parameter passed into split_matrix_iterate_QR().
    disentangle : bool, optional
        wether a disentangling routine should be executed before the second split. Default: False.
    disentangle_options : dictionary, optional
        options that get passed into the disentangler as kwargs.
        See src/utility/disentangle/disentangle.py for more information
    debug_dict : dictionary, optional
        dictionary in which debug information is saved. Default: None.

    Returns
    -------
    A : np.ndarray of shape (chi_1, D1, D2)
        resulting A tensor, isometry along (chi_1, (D1, D2))
    B : np.ndarray of shape (D1, chi_2, chi)
        resulting B tensor, isometry along ((D1, chi_2), chi)
    C : np.ndarray of shape (D2, chi, chi_3)
        resulting C tensor, normalized.
    """
    chi_1, chi_2, chi_3 = T.shape
    A = np.reshape(T, (chi_1, chi_2*chi_3))
    if N_iters_svd is None:
        A, theta = utility.split_matrix_svd(A, D1*D2)
    else:
        A, theta, _, _ = utility.split_matrix_iterate_QR(A, D1*D2, N_iters_svd, eps_svd)
    A = np.reshape(A, (chi_1, D1, D2))
    theta = np.reshape(theta, (D1, D2, chi_2, chi_3)) # (D1, D2), (chi_2, chi_3) -> D1, D2, chi_2, chi_3
    # Check if the tripartite decomposition can be performed without error, in which case we don't need to disentangle
    if D1*chi_2 == chi:
        disentangle = False
    # Optionally disentangle
    if disentangle:
        theta = theta.transpose(2, 0, 1, 3) # D1, D2, chi_2, chi_3 -> chi_2, D_1, D_2, chi_3
        U = disentangle_lib.disentangle(theta, debug_dict=debug_dict, chi=chi, **disentangle_options)
        # Apply U to W
        theta = np.tensordot(U, theta, ([2, 3], [1, 2])) # D1 D2 [D1*] [D2*]; chi_2 [D1] [D2] chi_3 -> D1 D2 chi_2 chi_3
        theta = theta.transpose((0, 2, 1, 3)) # D1, D2, chi_2, chi_3 -> D1, chi_2, D2, chi_3
        # Contract A with U^\dagger
        A = np.tensordot(A, np.conj(U), ([1, 2], [2, 3])) # chi_1 [D1] [D2]; D1* D2* [D1] [D2] -> chi_1 D1 D2
    else:
        theta = theta.transpose(0, 2, 1, 3) # D1, D2, chi_2, chi_3 -> D1, chi_2, D2, chi_3
    # Split the theta tensor into B and C
    theta = theta.reshape(D1*chi_2, D2*chi_3) # D1, chi_2, D2, chi_3 -> (D1, chi_2), (D2, chi_3)
    if N_iters_svd is None:
        B, C = utility.split_matrix_svd(theta, chi)
    else:
        B, C, _, _ = utility.split_matrix_iterate_QR(theta, chi, N_iters_svd, eps_svd)
    B = B.reshape(D1, chi_2, chi)
    C = C.reshape(chi, D2, chi_3)
    C = C.transpose(1, 0, 2) # chi, D2, chi_3 -> D2, chi, chi_3
    return A, B, C