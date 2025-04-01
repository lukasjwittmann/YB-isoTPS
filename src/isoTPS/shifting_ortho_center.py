import numpy as np
from ..utility import utility

def move_ortho_center_up(W_center, W, chi_max, min_dims=False, options={"mode" : "svd"}):
   """
   Moves the orthogonality center up:

            |                           |                                                        
            v                           v                                                           
            |                           |                                                            
      -->----W-----<--            -->-W_center-<--                                                    
            |                           |                                                          
            v             ->            ^                                                           
            |                           |                                                           
      -->-W_center-<--            -->----W-----<--                                                    
            |                           |                                                         
            ^                           ^                                                            
            |                           |                                                          

   Parameters
   ----------
   W_center : np.ndarray of shape (l_c, u_c, r_c, d_c)
      orthogonality center
   W : np.ndarray of shape (l, u, r, d)
      orthogonality hypersurface tensor right above the orthogonality center
   chi_max : int
      maximal bond dimension
   options : dict, optional
      options specifying how the splitting of the W_tensor is performed.
      See function split_matrix() in "src/utility/utility.py" for more details.
      In most cases, the splitting can be done with a simple QR decomposition, in which
      case the options dictionary doesn't matter (as both splitting modes choose the QR decomposition
      if possible).

   Returns
   -------
   W : np.ndarray of shape (l_c, u_c, r_c, d_c)
      orthogonality hypersurface tensor right below the new orthogonality center
   W_center : np.ndarray of shape (l, u, r, d)
      the new orthongonality center tensor
   """
   l_c, u_c, r_c, d_c = W_center.shape
   W_center = W_center.transpose(0, 2, 3, 1).reshape(l_c*r_c*d_c, u_c) # l, u, r, d -> l, r, d, u -> (l, r, d), u
   assert("mode" in options)
   mode = options["mode"]
   N_iters = None
   if "N_iters" in options:
      N_iters = options["N_iters"]
   if min_dims:
      chi = min(W_center.shape[0], W_center.shape[1], chi_max)
   else:
      chi = min(W_center.shape[0], chi_max)
   Q, R = utility.split_matrix(W_center, chi, mode, N_iters)
   W_center = np.tensordot(W, R, ([3], [1])) # l u r [d]; u_c [u_c*] -> l u r u_c = l u r d
   W = Q.reshape(l_c, r_c, d_c, -1).transpose(0, 3, 1, 2) # (l, r, d), u -> l, r, d, u -> l, u, r, d
   return W, W_center

def move_ortho_center_down(W, W_center, chi_max, min_dims=False, options={"mode" : "svd"}):
   """
   Moves the orthogonality center down:

            |                           |                                                        
            v                           v                                                           
            |                           |                                                            
      -->-W_center-<--            -->----W-----<--                                                    
            |                           |                                                          
            ^             ->            v                                                           
            |                           |                                                           
      -->----W-----<--            -->-W_center-<--                                                    
            |                           |                                                         
            ^                           ^                                                            
            |                           |                                                          

   Parameters
   ----------
   W : np.ndarray of shape (l, u, r, d)
      orthogonality hypersurface tensor right below the orthogonality center
   W_center : np.ndarray of shape (l_c, u_c, r_c, d_c)
      orthogonality center
   chi_max : int
      maximal bond dimension
   options : dict, optional
      options specifying how the splitting of the W_tensor is performed.
      See function split_matrix() in "src/utility/utility.py" for more details.
      In most cases, the splitting can be done with a simple QR decomposition, in which
      case the options dictionary doesn't matter (as both splitting modes choose the QR decomposition
      if possible).

   Returns
   -------
   W_center : np.ndarray of shape (l, u, r, d)
      the new orthongonality center tensor
   W : np.ndarray of shape (l_c, u_c, r_c, d_c)
      orthogonality hypersurface tensor right above the new orthogonality center
   """
   l_c, u_c, r_c, d_c = W_center.shape
   W_center = W_center.reshape(l_c * u_c * r_c, d_c) # l, u, r, d -> (l, u, r), d
   assert("mode" in options)
   mode = options["mode"]
   N_iters = None
   if "N_iters" in options:
      N_iters = options["N_iters"]
   if min_dims:
      chi = min(W_center.shape[0], W_center.shape[1], chi_max)
   else:
      chi = min(W_center.shape[0], chi_max)
   Q, R = utility.split_matrix(W_center, chi, mode, N_iters)
   W_center = np.tensordot(W, R, ([1], [1])).transpose(0, 3, 1, 2) # l [u] r d; d_c [d_c*] -> l r d d_c = l r d u -> l, u, r, d
   W = Q.reshape(l_c, u_c, r_c, -1) # (l, u, r), d -> l, u, r, d
   return W_center, W