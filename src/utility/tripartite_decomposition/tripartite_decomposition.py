import numpy as np
from .. import utility
from . import tripartite_decomposition_svd
from . import tripartite_decomposition_iterate_polar
from . import tripartite_decomposition_iterate_rank_reduce
from . import tripartite_decomposition_loop

def tripartite_decomposition(T, D1, D2, chi, mode="svd", debug_dict=None, **kwargs):
    """
    Computes the tripartite decomposition of tensor T:

                                              chi_3 /
                                                   /
                    chi_3 /                      (C) 
                         /                  D_2 / |
               chi_1    /             chi_1    /  |
                -->--(T)        ->     -->--(A)   ^ chi
                        \                      \  |
                         \                  D_1 \ |
                    chi_2 \                      (B)
                                                   \ 
                                              chi_2 \      

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
    mode : std, one of {"svd", "iterate_polar", "iterate_rank_reduce", "iterate_loop"}, optional
        string deciding the splitting algorithm. Default: "svd".
    debug_dict : dictionary, optional
        dictionary in which debug information is saved. Default: None.
    **kwargs
        remaining kwargs are passed into the respective method chosen with mode.
        See the different called functions for more information.

    Returns
    -------
    A : np.ndarray of shape (chi_1, D1, D2)
        resulting A tensor, isometry along (chi_1, (D1, D2))
    B : np.ndarray of shape (D1, chi_2, chi)
        resulting B tensor, isometry along ((D1, chi_2), chi)
    C : np.ndarray of shape (D2, chi, chi_3)
        resulting C tensor, normalized.
    """
    if mode == "svd":
        return tripartite_decomposition_svd.tripartite_decomposition(T, D1, D2, chi, debug_dict=debug_dict, **kwargs)
    elif mode == "iterate_polar":
        return tripartite_decomposition_iterate_polar.tripartite_decomposition(T, D1, D2, chi, debug_dict=debug_dict, **kwargs)
    elif mode == "iterate_rank_reduce":
        return tripartite_decomposition_iterate_rank_reduce.tripartite_decomposition(T, D1, D2, chi, debug_dict=debug_dict, **kwargs)
    elif mode == "iterate_loop":
        return tripartite_decomposition_loop.tripartite_decomposition(T, D1, D2, chi, debug_dict=debug_dict, **kwargs)
    else:
        temp = r'{"svd", "iterate_polar", "iterate_rank_reduce", "iterate_loop"}'
        raise NotImplementedError(f"A tripartite decomposition with mode \"{mode}\" is not implemented. Choose one of {temp}")