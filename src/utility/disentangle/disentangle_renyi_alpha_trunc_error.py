from . import disentangle_renyi_alpha
from . import disentangle_trunc_error
from .. import utility
import numpy as np

def disentangle(theta, chi, options_renyi, options_trunc, debug_dict=None):
    """
    disentangles the wavefunction tensor theta by first minimizing the renyi entropy and
    then minimizing the truncation error.

    Parameters
    ----------
    theta : np.ndarray of shape (l, i, j, r)
        wavefunction tensor to be disentangled.
    chi : int
        bond dimension that the SVD of Utheta is truncated to. Smaller chi will speed up the algorithm
        but generally lead to worse results.
    options_renyi : dict
        dictionary with options that get passed as kwargs to the renyi alpha disentangler
        See src/utility/disentangle/disentangle_renyi_alpha.py for more information.
    options_trunc : dict
        dictionary with options that get passed as kwargs to the trunc error disentangler.
        See src/utility/disentangle/disentangle_trunc_error.py for more information.

    Returns
    -------
    U_final : np.ndarray of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    _, d1, d2, _ = theta.shape
    debug_dict_renyi = None
    debug_dict_trunc_error = None
    if debug_dict is not None:
        debug_dict_renyi = {"debug_level": debug_dict["debug_level"]}
        debug_dict_trunc_error = {"debug_level": debug_dict["debug_level"]}
    # Disentangle with Renyi disentangler
    U = disentangle_renyi_alpha.disentangle(theta, debug_dict=debug_dict_renyi, **options_renyi)
    theta = np.tensordot(U, theta, ([2, 3], [1, 2])).transpose(2, 0, 1, 3) # i j [i*] [j*]; ml [d1] [d2] mr -> i j ml mr -> ml i j mr
    # Disentangle with truncation error disentangler
    U2 = disentangle_trunc_error.disentangle(theta, chi, debug_dict=debug_dict_trunc_error, **options_trunc)
    U = np.dot(U2.reshape(d1*d2, d1*d2), U).reshape(d1, d2, d1, d2)
    # save debug information
    if debug_levels.check_debug_level(debug_dict, debug_levels.DebugLevel.LOG_DISENTANGLER_TRIPARTITE_DECOMPOSITION_ITERATION_INFO):
        for key, value in debug_dict_renyi.items():
            utility.append_to_dict_list(debug_dict, key + "_renyi", value)
        for key, value in debug_dict_renyi.items():
            utility.append_to_dict_list(debug_dict, key + "_trunc_error", value)
    return U

def disentangle_approx(theta, chi, options_renyi, options_trunc, debug_dict=None):
    """
    disentangles the wavefunction tensor theta by first minimizing the approximate renyi entropy 
    and then minimizing the approximate truncation error.

    Parameters
    ----------
    theta : np.ndarray of shape (l, i, j, r)
        wavefunction tensor to be disentangled.
    chi : int
        bond dimension that the SVD of Utheta is truncated to. Smaller chi will speed up the algorithm
        but generally lead to worse results.
    options_renyi : dict
        dictionary with options that get passed as kwargs to the renyi alpha disentangler
        See src/utility/disentangle/disentangle_renyi_alpha.py for more information.
    options_trunc : dict
        dictionary with options that get passed as kwargs to the trunc error disentangler.
        See src/utility/disentangle/disentangle_trunc_error.py for more information.

    Returns
    -------
    U_final : np.ndarray of shape (i, j, i*, j*)
        final disentangling unitary after optimization
    """
    _, d1, d2, _ = theta.shape
    debug_dict_renyi = None
    debug_dict_trunc_error = None
    if debug_dict is not None:
        debug_dict_renyi = {"debug_level": debug_dict["debug_level"]}
        debug_dict_trunc_error = {"debug_level": debug_dict["debug_level"]}
    # Disentangle with Renyi disentangler
    U = disentangle_renyi_alpha.disentangle_approx(theta, chi, debug_dict=debug_dict_renyi, **options_renyi)
    theta = np.tensordot(U, theta, ([2, 3], [1, 2])).transpose(2, 0, 1, 3) # i j [i*] [j*]; ml [d1] [d2] mr -> i j ml mr -> ml i j mr
    # Disentangle with truncation error disentangler
    U2 = disentangle_trunc_error.disentangle_approx(theta, chi, debug_dict=debug_dict_trunc_error, **options_trunc)
    U = np.dot(U2.reshape(d1*d2, d1*d2), U).reshape(d1, d2, d1, d2)
    # save debug information
    if debug_levels.check_debug_level(debug_dict, debug_levels.DebugLevel.LOG_DISENTANGLER_TRIPARTITE_DECOMPOSITION_ITERATION_INFO):
        for key, value in debug_dict_renyi.items():
            utility.append_to_dict_list(debug_dict, key + "_renyi", value)
        for key, value in debug_dict_renyi.items():
            utility.append_to_dict_list(debug_dict, key + "_trunc_error", value)
    return U