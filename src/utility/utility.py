import numpy as np
import scipy
import scipy.linalg
#import hdfdict

"""
This file implements several utility functions that are used throughout the code base
"""

def safe_svd(A, full_matrices=True):
    """
    Computes the Singular Value Decomposition A = U@S@V. If the numpy svd does not converge,
    scipy's SVD with the less efficient but more involved general rectangular approach is used,
    which is more likely to converge.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        The matrix that should be decomposed using SVD.
    full_matrices : bool, optional
        determines the shape of the output matrices. See official
        numpy documentation for more details.
    
    Returns
    -------
    U : np.ndarray of shape (n, chi)
        isometric matrix. A = U@np.diag(S)@V.
    S : np.ndarray of shape (chi, )
        vector containing the real singular values >= 0. A = U@np.diag(S)@V.
    V : np.ndarray of shape (chi, m)
        V.T is an isometric matrix. A = U@np.diag(S)@V.
    """
    try:
        return np.linalg.svd(A, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        if np.isnan(A).any() or np.isinf(A).any():
            print("[WARNING]: Trying to perform SVD on a matrix with nan or inf entries!")
        U, S, V = scipy.linalg.svd(A, full_matrices=full_matrices, lapack_driver='gesvd')
        if np.isnan(U).any() or np.isinf(U).any() or np.isnan(S).any() or np.isinf(S).any() or np.isnan(V).any() or np.isinf(V).any():
            print("[WARNING] scipy SVD did not converge!")
            m, n = A.shape
            k = min(m, n)
            return np.zeros(m, k), np.zeros(k), np.zeros(k, n)
        return U, S, V

def split_and_truncate(A, chi_max=0, eps=0):
    """
    Performs an SVD of the matrix A and truncates the singular values to the bond dimension chi_max.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        The matrix that should be split using SVD.
    chi_max : int, optional
        The maximum bond dimension to which the result is truncated. If this is set to zero, the algorithm
        acts as if there is no maximum bond dimension. Default: 0.
    eps : float, optional
        All singular values smaller than eps are truncated. Default: 0.

    Returns
    -------
    U : np.ndarray of shape (n, chi)
        isometric matrix. A \approx U@np.diag(S)@V.
    S : np.ndarray of shape (chi, )
        vector containing the normalized real singular values >= 0. A \approx U@np.diag(S)@V.
    V : np.ndarray of shape (chi, m)
        V.T is an isometric matrix. A \approx U@np.diag(S)@V.
    norm : float
        the norm of the unnormalized (but already truncated) singular values, np.linalg.norm(S).
    error : float
        the error of the truncation (sum of the square of all singular values being thrown away).
    """
    # perform SVD
    U, S, V = safe_svd(A, full_matrices=False)
    # truncate
    if chi_max > 0:
        chi_new = min(chi_max, np.sum(S >= eps))
    else:
        chi_new = np.sum(S>=eps)
    assert chi_new >= 1
    piv = np.argsort(S)[::-1][:chi_new]  # keep the largest chi_new singular values
    error = np.sum(S[chi_new:]**2)
    if error > 1.e-1:
        print("[WARNING]: larger error detected in SVD, error =", error, "sum of remaining singular values:", np.sum(S[:chi_new]**2))
    U, S, V = U[:, piv], S[piv], V[piv, :]
    # renormalize
    norm = np.linalg.norm(S)
    if norm < 1.e-7 and norm != 0.0:
        print(f"[WARNING]: Small singular values, norm(S) = {norm}")
    if norm != 0.0:
        S = S / norm
    return U, S, V, norm, error
    
def random_isometry(N, M):
    """
    Returns a random (N, M) complex isometry.

    Parameters
    ----------
    N, M : int
        shape of the isometry
    
    Returns
    -------
    W : np.ndarray of shape (N, M)
        random isometry
    """
    z = (np.random.random((N, N)) + 1j*np.random.random((N, N)))/np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d/np.abs(d)
    q = q@np.diag(ph)@q
    return q[:, :M]

def random_unitary(N):
    """
    Returns a random (N, N) unitary drawn from the Haar measure
    
    Parameters
    ----------
    N: int
        dimension of the unitary
    
    Returns
    -------
    U : np.ndarray of shape (N, N)
        random unitary
    """
    from scipy.stats import unitary_group
    return unitary_group.rvs(N)

def check_isometry(A):
    """
    Cecks if the given matrix A of shape (in, out) is in fact an isometry, ie. if
    A^\dagger A = 1 and if P = A A^\dagger is a projector, ie. P^2 = P.
    Returns True on success and False on failure.

    Parameters
    ----------
    A : np.ndarray of shape (in, out)
        the matrix that is to be checked

    Returns
    -------
    result : bool
        wether A is an isometry or not
    """
    AA_dagger = A@np.conj(A.T)
    return np.all(np.isclose(np.conj(A.T)@A, np.eye(A.shape[1]))) and np.all(np.isclose(AA_dagger, AA_dagger@AA_dagger))

def flip_W(W):
    """
    Flips the given W tensor along the vertical axis

         u                   u
         |                   |
     l--(W)--r    <->    r--(W)--l
         |                   |
         d                   d

    Parameters
    ----------
    W : np.ndarray with ndim = 4 or None
        the W tensor to be flipped

    Returns
    -------
    W_prime : np.ndarray with ndim = 4 or None
        the flipped W tensor, or None if W == None
    """
    if W is None: 
        return None
    return np.transpose(W, (2, 1, 0, 3)) # l, u, r, d <-> r, u, l, d

def flip_T_square(T):
    """
    Flips the given T tensor (square isoTPS) along the vertical axis

     lu     ru          ru     lu     
      \  p  /            \     /     
       \ | /              \   /    
        (T)       <->      (T)  
       /   \              /   \     
      /     \            /     \    
     ld     rd          rd     ld

    Parameters
    ----------
    T : np.ndarray with ndim = 5 or None
        the T tensor to be flipped

    Returns
    -------
    T_prime : np.ndarray with ndim = 5 or None
        the flipped T tensor, or None if T == None
    """
    if T is None:
        return None
    return np.transpose(T, (0, 4, 3, 2, 1)) # p, ru, rd, ld, lu <-> p, lu, ld, rd, ru

def flip_T_honeycomb(T):
    """
    Flips the given T tensor (honeycomb isoTPS) along the vertical axis

         p  ru           ru  p           lu  p                   
         | /               \ |             \ |                   
     l--(T)       <->       (T)--l   =      (T)--r                    
           \               /               /                     
            rd           rd              ld  

    Parameters
    ----------
    T : np.ndarray with ndim = 4 or None
        the T tensor to be flipped

    Returns
    -------
    T_prime : np.ndarray with ndim = 4 or None
        the flipped T tensor, or None if T == None        
    """
    if T is None:
        return None
    return T.transpose(0, 3, 2, 1) # p, ru, rd, l <-> p, r, ld, lu

def flip_onesite_square(T, W, Wp1):
    """
    Flips all the tensors of the one-site wave function (square isoTPS) along the vertical axis
    """
    return flip_T_square(T), flip_W(W), flip_W(Wp1)

def flip_twosite_square(T1, T2, Wm1, W, Wp1):
    """
    Flips all the tensors of the two-site wave function (square isoTPS) along the vertical axis
    """
    return flip_T_square(T1), flip_T_square(T2), flip_W(Wm1), flip_W(W), flip_W(Wp1)

def flip_onesite_honeycomb(T, W, Wp1):
    """
    Flips all the tensors of the one-site wave function (honeycomb isoTPS) along the vertical axis
    """
    return flip_T_honeycomb(T), flip_W(W), flip_W(Wp1)

def flip_twosite_honeycomb(T1, T2, Wm1, W, Wp1):
    """
    Flips all the tensors of the one-site wave function (honeycomb isoTPS) along the vertical axis
    """
    return flip_T_honeycomb(T1), flip_T_honeycomb(T2), flip_W(Wm1), flip_W(W), flip_W(Wp1)

def flip_twosite_op(op):
    """
    Flips the given two site operator along the vertical axis

       i*    j*                j*    i*             
       |     |                 |     |             
    |-----------|           |-----------|          
    |    op     |    <->    |    op     |
    |-----------|           |-----------|          
       |     |                 |     |          
       i     j                 j     i          

    Parameters
    ----------
    op : np.ndarray with ndim = 4 or None
        the op tensor to be flipped

    Returns
    -------
    op_prime : np.ndarray with ndim = 4 or None
        the flipped op tensor, or None if op == None
    """
    if op is None:
        return None
    return np.transpose(op, (1, 0, 3, 2)) # i, j, i*, j* -> j, i, j*, i*

def lq(X):
    """
    Performs an LQ decomposition of the given matrix

    Parameters
    ----------
    X: np.ndarray of shape (n, m)
        the matrix of which the LQ decomposition is taken
    
    Returns
    -------
    L: np.ndarray of shape (n, chi)
        L factor of the LQ decomposition.
    Q: np.ndarray of shape (chi, m)
        Q factor of the LQ decomposition. Q.T is an isometry.
    """
    Q, R = np.linalg.qr(X.T)
    return R.T, Q.T

def split_dims(chi, D_max):
    """
    This function tries to find integers D1, D2 > 0 such that |chi - D1*D2| is as small as possible and D1, D2 <= D_max.
    for the best achievable distance |D1*D2 - chi| the function also tries to split D1 and D2 as evenly as possible.
    It holds D1 <= D2.

    Parameters
    ----------
    chi : int
        integer to be split
    D_max : int
        maximal bond dimension

    Returns
    -------
    D1, D2 : int, int:
        best splitting found by the algorithm. It holds D1*D2 <= chi; D1, D2 <= D_max and D1 <= D2.
    """
    if chi >= D_max**2:
        return D_max, D_max
    best_prod = 1
    best_sum = 2
    best_D1 = 1
    best_D2 = 1
    for D1 in range(1, int(np.floor(np.sqrt(chi))) + 1):
        D2 = min(chi // D1, D_max)
        if D1 * D2 > best_prod:
            best_prod = D1 * D2
            best_sum = D1 + D2
            best_D1, best_D2 = D1, D2
        elif D1 * D2 == best_prod and D1 + D2 < best_sum:
            best_sum = D1 + D2
            best_D1, best_D2 = D1, D2
    return best_D1, best_D2

def split_matrix_svd(A, chi):
    """
    Splits a (n x m) matrix A into a (n x chi) isometry B and a (chi x m) matrix C,
    using Singular Value Decomposition. This function asssumes chi <= n.
    If chi >= min(n, m), the decomposition is numerically exact and the QR decomposition is used, 
    because it is faster than the SVD in practice.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        the normalized matrix to be split.
    chi : int
        split dimension. should be >= 1.

    Returns
    -------
    B : np.ndarray of shape (n, chi)
        first factor of the split. B is an isometry. A \approx B@C.
    C : np.ndarray of shape (chi, m)
        second factor of the split. A \approx B@C.
    """
    norm = np.linalg.norm(A)
    assert np.isclose(norm, 1., atol=1.e-8), "the matrix must be normalized."
    n, m = A.shape
    assert 1 <= chi <= n, f"for ({n} x {m}) matrix, chi must be between 1 and {n}."
    if chi >= min(n, m):
        if chi > m:  
            Q, R = np.linalg.qr(A, mode="complete")  
            return Q[:, :chi], R[:chi, :]
        else:
            Q, R = np.linalg.qr(A, mode="reduced") 
            return Q, R
    # Split and truncate A via SVD
    B, S, V = safe_svd(A, full_matrices=False)
    piv = np.argsort(S)[::-1][:chi]
    B, S, V = B[:, piv], S[piv], V[piv, :]
    # Renormalize
    S /= np.linalg.norm(S)
    # Isometrize B
    B, R = np.linalg.qr(B)
    # Absorb R and S into V to form C
    C = R @ np.diag(S) @ V
    return B, C

def split_matrix_iterate_QR(A, chi, N_iters, eps=1e-9, C0=None, smart_initial_condition=True, normalize=True, log_iterates=False):
    """
    Splits a (n x m) matrix A into a (n x chi) isometry B and a (chi x m) matrix C,
    using N_iters iterations of a sweeping algorithm using only QR decompositions and matrix products.
    Per iteration 2 QR decompositions and 2 matrix products are computed.
    This function asssumes chi <= max(n, m). If chi >= min(n, m), a single QR decomposition suffices
    to compute the numerically exact solution.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        the matrix to be split.
    chi : int
        split dimension. should be >= 1.
    N_iters : int
        maximum number of iterations
    eps : float, optional
        if the relative decrease of the error after one iteration is smaller than eps,
        the algorithm terminates. Default value: 1e-9.
    C0 : np.ndarray or None, optional
        Initialization for the C matrix. If multiple splits are executed on similar matrices,
        the result of a previous split can be a very good initialization for the next split.
        Default value: None
    smart_initial_condition: bool, optional
        Determines the initialization of C, if C0 is None.
        If this is set to True, the C matrix is initialized by a reordered slicing of A.
        If this is set to False, the C matrix is initialized with identity.
        Default value: True
    normalize : bool, optional
        Determines wether the C matrix is to be normalized. Default: True
    log_iterates : bool, optional
        If this is set to True, the iterates are stored in a list and returned. Default: False.

    Returns
    -------
    B : np.ndarray of shape (n, chi)
        first factor of the split. B is an isometry. A \approx B@C.
    C : np.ndarray of shape (chi, m)
        second factor of the split. A \approx B@C.
    num_iters : int
        the number of iterations used
    iterates : List of (np.ndarray, np.ndarray) or None
        List of iterates (B_i, C_i). If log_iterates is set to False,
        None is returned instead.
    """
    assert(chi > 0)
    iterates = None
    if log_iterates:
        iterates = []
    if chi is None or chi == min(A.shape[0], A.shape[1]):
        Q, R = np.linalg.qr(A)
        Q, R = np.ascontiguousarray(Q), np.ascontiguousarray(R)
        if log_iterates:
            iterates.append((Q, R))
        return Q, R, 0, iterates
    elif chi > min(A.shape[0], A.shape[1]):
        Q, R = np.linalg.qr(A)
        Q = np.ascontiguousarray(Q) @ np.eye(Q.shape[1], chi, dtype=Q.dtype)
        Q, R2 = np.linalg.qr(Q)
        R = np.ascontiguousarray(R2) @ np.eye(chi, R.shape[0], dtype=R.dtype) @ R
        Q, R = np.ascontiguousarray(Q), np.ascontiguousarray(R)
        if log_iterates:
            iterates.append((Q, R))
        return Q, R, 0, iterates
    assert(N_iters > 0)
    if C0 is not None:
        C = C0
    elif smart_initial_condition:
        # find the chi largest rows
        temp = np.sum(np.abs(A), 1)
        piv = np.argsort(temp)[::-1][:chi]
        # slice A matrix
        C = A[piv, :]
    else:
        # Initialize C with identity
        C = np.eye(chi, A.shape[1], dtype=A.dtype)
    error = None
    for n in range(N_iters):
        # Isometrize C
        C, _ = np.linalg.qr(C.T)
        # Compute B
        B = np.dot(A, np.conj(C))
        # isometrize B
        B, _ = np.linalg.qr(B)
        # Compute C
        C = np.dot(np.conj(B).T, A)
        # Store iterates
        if log_iterates:
            iterates.append((B, C))
        # Check if we are done
        error_new = np.linalg.norm(A - B@C)
        if error is not None and (np.isclose(error_new, 0) or np.abs((error - error_new)/error) < eps):
            break
        error = error_new
    if normalize:
        return B, C / np.linalg.norm(C), n + 1, iterates
    else:
        return B, C, n + 1, iterates

def split_matrix(A, chi, mode, N_iters=None):
    """
    Splits a (n x m) matrix A into a (n x chi) isometry B and a (chi x m) matrix C.
    Depending on the selected mode, either the function split_matrix_svd() or
    split_matrix_iterate_QR() is called for the splitting.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        the matrix to be split.
    chi : int
        split dimension. should be >= 1.
    mode : str, one of {"svd", "iterate"}
        used for selecting the splitting mode.
    N_iters : int or None, optional
        maximum number of iterations. Only used when mode == "iterate". Default: None.

    Returns
    -------
    B : np.ndarray of shape (n, chi)
        first factor of the split. B is an isometry. A \approx B@C.
    C : np.ndarray of shape (chi, m)
        second factor of the split. A \approx B@C.
    """
    if mode == "svd":
        return split_matrix_svd(A, chi)
    elif mode == "iterate":
        B, C, _, _ = split_matrix_iterate_QR(A, chi, N_iters)
        return B, C
    else:
        raise NotImplementedError(f"split_matrix is not implemted for mode = {mode}")

def isometrize_polar(A):
    """
    Finds the isometry B, which has the same shape as A and minimizes the distance |A - B|, using a polar decomposition.
    It is assumed that A is an (n, m) matrix with n >= m. The polar decomposition is implemented using an SVD.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        real or complex matrix

    Returns
    -------
    B : np.ndarray of shape (n, m)
        isometry closest to A
    """
    U, _, V = safe_svd(A, full_matrices=False)
    return U@V

def calc_U_bonds(H_bonds, dt):
    """
    Given the Hamiltonian H as a list of two-site operators,
    calculate expm(-dt*H). Note that no imaginary 'i' is included,
    thus real 'dt' means imaginary time evolution!

    Parameters
    ----------
    H_bonds : List of np.ndarray with ndim = 4
        list of two-site operators making up the Hamiltonian.
    dt : complex
        real or imaginary time. Note that real dt means imaginary time evolution.

    Returns
    -------
    U_bonds : List of np.ndarray with ndim = 4
        list of two-site real or imaginary time evolution operators
    """
    U_bonds = []
    d = H_bonds[0].shape[0]
    for H in H_bonds:
        if H is None:
            U_bonds.append(None)
        else:
            H = np.reshape(H, (d*d, d*d))
            U = scipy.linalg.expm(-dt*H)
            U_bonds.append(np.reshape(U, (d, d, d, d)))
    return U_bonds

def compute_op_list(L, op):
    """
    Given a single site operator, computes a list of operators
    eye x eye x ... x eye x op x eye x ... x eye,
    where the ith entry of the list puts the operator on site i

    Parameters
    ----------
    L : int
        number of sites in the chain
    op : np.ndarray of size (d, d)
        single-site operator

    Returns
    -------
    result : List of np.ndarray of size (d^L, d^L)
        list of single site operators acting on the full Hilbert space
    """
    result_list = []
    eye = np.eye(*op.shape)
    for j in range(L):
        result = eye
        if j == 0:
            result = op
        else:
            for i in range(1, j):
                result = scipy.sparse.kron(result, eye)
            result = scipy.sparse.kron(result, op)
        for i in range(j+1, L):
            result = scipy.sparse.kron(result, eye)
        result_list.append(result)
    return result_list

def average_site_expectation_value(L, psi, op):
    """
    Given a wave function psi, the number of sites L, and an operator op,
    this computes the average site expectation value of op:
    1/L sum_{i=0}^{i=L-1} <psi| eye x eye x ... x eye x op_i x eye x ... x eye |psi>

    Parameters
    ----------
    L : int
        number of sites
    psi : np.ndarray of size (L^d, )
        vector representing the wave function
    op : np.ndarray of size (d, d)
        single-site operator

    Returns
    -------
    result : float
        average site expectation value
    """
    op_list = compute_op_list(L, op)
    result = 0.
    for op in op_list:
        result += (psi.T@op@psi)/(psi.T@psi)
    return result / L

def append_to_dict_list(d, key, value):
    """
    appends an object to the list d[key], if the key is already in the dictionary,
    else created a new list d[key] = [value].

    Parameters
    ----------
    d : Dict
        dictionary
    key : str
        key into the dictionary. if key in d, d[key] must be a list.
    value : Any
        value that is appended to the list d[key]. If value itself is a list, it is
        not appended but concatenated to d[key].
    """
    if key in d:
        d[key] = list(d[key])
        if type(value) == list:
            d[key] += value
        else:
            d[key].append(value)
    else:
        if type(value) == list:
            d[key] = value
        else:
            d[key] = [value]

def hdf_dict_to_python_dict(d):
    """
    Creates a python dictionary from a hdf dictionary loaded with hdfdict.load

    Parameters
    ----------
    d : hdfdict
        the hdfdict that we want to turn into a python dictionary
    
    Returns
    -------
    result : dict
        resulting python dictionary
    d = dict(d)
    for key, value in d.items():
        if isinstance(value, hdfdict.hdfdict.LazyHdfDict):
            d[key] = hdf_dict_to_python_dict(value)
        elif isinstance(value, bytes):
            d[key] = value.decode()
    return d
    """

def load_dict_from_file(filename):
    """
    return hdf_dict_to_python_dict(hdfdict.load(filename))
    """

def turn_lists_to_dicts(d):
    """
    Recursively loops through the contents of the given dictionary, turning all lists [element_1, element_2, ..., element_n] 
    into dictionaries {"1": element_1, "2": element_2, ..., "n": element_n} if element_1, element_2, ..., element_n are lists
    or dicts themselves. This is used for storing lists of elements in 
    h5-files using hdfdict.dump.

    Parameters
    ----------
    d : dict
        the dictionary that should be processed.
    """
    for key in d:
        if type(d[key]) is list:
            # Check if the elements are all of numerical type
            is_dict_or_list = False
            for element in d[key]:
                if type(element) == dict or type(element) == list:
                    is_dict_or_list = True
                    break
            if is_dict_or_list:
                # Turn list into dict
                temp_dict = {}
                for i, item in enumerate(d[key]):
                    temp_dict[str(i)] = item
                d[key] = temp_dict
        if type(d[key]) is dict:
            turn_lists_to_dicts(d[key])

def turn_dicts_to_lists(d):
    """
    Recursively loops through the contents of the given dictionary, turning all dictionaries of the form
    {"1": element_1, "2": element_2, ..., "n": element_n} back to lists lement_1, element_2, ..., element_n].
    This is used for recovering the original structure of the debug_log when loading from file.
    
    Parameters
    ----------
    d : dict
        the dictionary that should be processed.
    """
    # Go through all keys in d
    for key in d:
        if type(d[key]) is dict:
            # Check if d[key] has the expected list format
            i = 0
            while True:
                if str(i) in d[key]:
                    i += 1
                else:
                    i -= 1
                    break
            if i == -1 or len(d[key]) != i+1:
                # The dictionary is not in the expected format. Recursively go down if the element is a dictionary
                turn_dicts_to_lists(d[key])
            else:
                # The dictionary is in the expected format. Turn into a list!
                temp_list = []
                for j in range(i+1):
                    temp_list.append(d[key][str(j)])
                d[key] = temp_list
                # Go through the list and recursively call turn_dicts_to_lists if the elements are dictionaries themselves
                for j in range(i+1):
                    if type(d[key][j]) is dict:
                        turn_dicts_to_lists(d[key][j])


def get_flipped_As(As):
    """For a list of site tensors As, reverse the order and flip every tensor A vertically.

    2       4      1       3
     \  0  /        \  0  /
      \ | /          \ | / 
       (A)     ->     (A)
      /   \          /   \
     /     \        /     \
    1       3      2       4

    p ld lu rd ru -> p lu ld ru rd
    """
    if As is not None:
        return [np.transpose(A, (0, 2, 1, 4, 3)) for A in As[::-1]]
    else:
        return None

def get_flipped_hs(hs):
    """For a list of mpo tensors hs, reverse the order and flip every tensor h vertically.

         1           0
     2  /        2  /
     | /         | /
    (h)     ->  (h)
     | \         | \
     3  \        3  \
         0           1

    rd ru p p* -> ru rd p p*
    """
    if hs is not None:
        return [np.transpose(h, (1, 0, 2, 3)) for h in hs[::-1]]
    else:
        return None

def get_flipped_Cs(Cs):
    """For a list of orthogonality column or boundary mps tensors Cs, reverse the order and flip 
    every tensor C vertically.

         3                0
         |                |
         |                |
    1---(C)---2  ->  1---(C)---2
         |                |
         |                |
         0                3

    d l r u -> u l r d
    """
    if Cs is not None:
        return [np.transpose(C, (3, 1, 2, 0)) for C in Cs[::-1]]
    else:
        return None