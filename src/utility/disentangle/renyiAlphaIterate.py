import numpy as np
from .. import utility

"""
This file implements several iterate classes that are used in the disentangle process optimizing the renyi alpha entropy.
The optimization algorithm uses instances of iterate classes to represent the current iterate. Iterate classes must implement functionality
to evaluate the cost function, compute the gradient, and optionally compute the hessian vector product (if the iterate class is to be used in
the trust region optimizer). Iterate classes may use caching to minimize the number of necessary computations.
"""

class RenyiAlphaIterate:
    """
    Base class for the more specialized iterate classes.
    """

    def __init__(self, U, theta, alpha=0.5, chi_max=None, N_iters_svd=None, eps_svd=0.0, old_iterate=None):
        """
        Initializes new iterate.

        Parameters
        ----------
        U : np.ndarray of shape (i, j, i*, j*)
            disentangling unitary.
        theta : np.ndarray of shape (l, i, j, r)
            wavefunction tensor to be disentangled.
        alpha : float, optional
            renyi alpha. Default: 0.5.
        chi_max : int or None, optional 
            maximal bond dimension when splitting Utheta (approximate computation of cost function
            and gradient). If this is set to None, Utheta is not split. Default: None.
        N_iters_svd : int or None, optional
            number of iterations the qr splitting algorithm is run for approximating the SVD.
            If this is set to None, a full SVD is performed instead. Default: None.
        eps_svd : float, optional
            eps parameter passed into split_matrix_iterate_QR(), 
            see src/utility/utility.py for more information. Default: 0.0.
        old_iterate : element of RenyiAlphaIterate class or None, optional
            old iterate. This is used here for initializing the qr splitting algorithm with the
            old result, which can lead to faster convergence. Default: None.
        """
        self.U = U
        self.theta = theta
        self.alpha = alpha
        self.chi_max = chi_max
        self.N_iters_svd = N_iters_svd
        self.eps_svd = eps_svd
        self.l, self.D1, self.D2, self.r = self.theta.shape
        self.k = min(self.l*self.D1, self.D2*self.r)
        if self.chi_max is not None:
            self.k = min(chi_max, self.k)
        self.Y0 = None
        if self.N_iters_svd is not None and old_iterate is not None:
            self.Y0 = old_iterate.Y0

    def get_iterate(self):
        """
        Returns the actual iterate (the disentangling unitary of shape (i, j, i*, j*)).

        Returns
        -------
        U : np.ndarray of shape (i, j, i*, j*)
            disentangling unitary.
        """
        return self.U

class RenyiAlphaIterateCG(RenyiAlphaIterate):
    """
    Class representing a single iterate for the renyi-alpha conjugate gradients optimizer.
    Is used by both the full and the approximate Conjugate Gradients disentangler.
    Cashes important results such that they do not need to be computed multiple times.
    """

    def evaluate_cost_function(self):
        """
        Computes the value of the renyi alpha entropy 1/(1 - alpha) * log(sum_{i} s_i^(2*alpha)), with s_i the ith singular value
        of U@theta. If self.chi_max is not None, the result will only be approximate.

        Returns
        -------
        cost : float
            value of the cost function
        """
        # Compute Utheta
        Utheta = np.dot(np.ascontiguousarray(self.U).reshape((self.D1*self.D2, -1)), np.ascontiguousarray(self.theta.transpose(1, 2, 0, 3)).reshape(self.D1*self.D2, -1)) # l, i, j, r -> i, j, l, r -> (i, j), (l, r)
        Utheta = np.ascontiguousarray(np.ascontiguousarray(Utheta).reshape((self.D1, self.D2, self.l, self.r)).transpose((2, 0, 1, 3))).reshape((self.l*self.D1, -1)) # (i, j), (l, r) -> i, j, l, r -> l, i, j, r -> (l, i), (j, r)
        # Perform SVD
        if self.N_iters_svd is None:
            self.X, self.S, self.Y = utility.safe_svd(Utheta, full_matrices=False) # { D^9 }
            if self.chi_max is not None:
                idx = np.argsort(self.S)[::-1][:self.chi_max]
                self.X, self.S, self.Y = self.X[:, idx], self.S[idx], self.Y[idx, :]
        else:
            self.X, self.Y, _, _ = utility.split_matrix_iterate_QR(Utheta, self.chi_max, self.N_iters_svd, self.eps_svd, C0=self.Y0, normalize=False) # { N_iters_svd * D^7 }
            self.Y0 = self.Y
            XX, self.S, self.Y = np.linalg.svd(self.Y, full_matrices=False) # { D^5 }
            self.X = self.X@XX
        # Compute cost function
        self.S2alpha = np.sum(self.S**(2*self.alpha))
        return 1/(1-self.alpha)*np.log(np.real(self.S2alpha))

    def compute_gradient(self):
        """
        Computes the gradient of the renyi alpha entropy, projected to the tangent space of the iterate. 
        If self.chi_max is not None, the result will only be approximate.

        Returns
        -------
        grad : np.ndarray of shape (i, j, i*, j*)
            gradient of the cost function
        """
        S_prime = self.S**(2*self.alpha-1)
        grad = np.ascontiguousarray(self.X@np.diag(S_prime).astype(self.theta.dtype)@self.Y) # -> (l, i), (j, r)
        grad = np.reshape(grad, (self.l, self.D1, self.D2, self.r))
        grad = np.ascontiguousarray(grad.transpose((0, 3, 1, 2))).reshape(self.l*self.r, -1) # l, i, j, r -> l, r, i, j -> (l, r), (i, j)
        theta = np.ascontiguousarray(self.theta.transpose((0, 3, 1, 2))).reshape(self.l*self.r, -1) # l, i, j, r -> l, r, i, j -> (l, r), (i, j)
        grad = np.dot(grad.T, np.conj(theta)) # (i j) [(l r)]; [(l r)*] (i j)*
        grad = 2*self.alpha/(1-self.alpha)/self.S2alpha * grad
        # project gradient to tangent space
        U = self.U.reshape(self.D1*self.D2, -1)
        return (grad - 0.5 * U@(U.conj().T@grad + grad.conj().T@U)).reshape(self.D1, self.D2, self.D1, self.D2)

class RenyiAlphaIterateTRM(RenyiAlphaIterate):
    """
    Class representing a single iterate of the renyi-alpha trust region optimizer.
    Cashes important results such that they do not need to be computed multiple times.
    This is especially useful when multiple hessian vector products must be computed.
    """

    def __init__(self, U, theta, alpha=0.5):
        """
        Initializes new iterate.

        Parameters
        ----------
        See class RenyiAlphaIterate.
        """
        RenyiAlphaIterate.__init__(self, U, theta, alpha, chi_max=None, N_iters_svd=None, eps_svd=0, old_iterate=None)
        self.computed_gradient = False
        self.computed_hessian = False

    def evaluate_cost_function(self):
        """
        Computes the value of the renyi alpha entropy 1/(1 - alpha) * log(sum_{i} s_i^(2*alpha)), with s_i the ith singular value
        of U@theta.

        Returns
        -------
        cost : float
            value of the cost function
        """
        # Compute Utheta
        self.Utheta = np.tensordot(self.U, self.theta, ([2, 3], [1, 2])) # i j [i*] [j*]; l [i] [j] r -> i j l r { D^8 }
        self.Utheta = self.Utheta.transpose(2, 0, 1, 3) # i, j, l, r -> l, i, j, r
        # Perform SVD
        l, i, j, r = self.Utheta.shape
        self.X, self.S, self.Y = utility.safe_svd(self.Utheta.reshape(l*i, j*r), full_matrices=False) # { D^9 }
        self.Y = np.conj(self.Y.T)
        # Cache helper variables
        self.S2alpha = np.sum(self.S**(2*self.alpha))
        # Compute and return renyi-alpha entropy
        return 1/(1-self.alpha)*np.log(np.real(self.S2alpha))

    def compute_gradient(self):
        """
        Computes the gradient of the renyi alpha entropy, projected to the tangent space of the iterate.

        Returns
        -------
        grad : np.ndarray of shape (i, j, i*, j*)
            gradient of the cost function
        """
        if not self.computed_gradient:
            self.S_prime = self.S**(2*self.alpha-1)
            self.XSprimeY = (self.X*self.S_prime)@np.conj(self.Y.T)
            self.XSprimeY = self.XSprimeY.reshape(self.Utheta.shape)
            self.XSprimeYtheta = np.tensordot(self.XSprimeY, np.conj(self.theta), ([0, 3], [0, 3])) # [l] i j [r]; [l*] k* l* [r*] -> i j k* l*
            self.grad = np.tensordot(np.conj(self.XSprimeYtheta), self.U, ([2, 3], [2, 3])) # i' j' [k'] [l']; i j [k'] [l'] -> i' j' i j
            self.grad = np.tensordot(self.grad, self.U, ([0, 1], [0, 1])) # [i'] [j'] i j; [i'] [j'] k l -> i j k l
            self.grad = self.XSprimeYtheta - self.grad
            self.computed_gradient = True
        return self.alpha / (1 - self.alpha) / self.S2alpha * self.grad
    
    def compute_hessian_vector_product(self, dU):
        """
        Computes the riemannian hessian vector product of dU, which is the projection of D(grad f(U))[dU] to the tangent space of U. 
        When this is called for the first time, several intermediate results are cached for the case that this is called again in the future.

        Parameters
        ----------
        dU : np.ndarray of shape (i, j, i*, j*)
            element from the tangent space of U

        Returns
        -------
        hvp : np.ndarray of shape (i, j, i*, j*)
            hessian vector product H@dU
        """
        if not self.computed_hessian:
            # Cache helper results
            self.F = np.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(self.k):
                    if i != j:
                        if self.S[i] == self.S[j]:
                            self.F[i, j] = 0.0
                        else:
                            self.F[i, j] = 1/(self.S[j]**2 - self.S[i]**2)
            # Avoid dividing by zero
            non_zero_indices = np.where(self.S != 0.0)
            self.S_prime_prime = np.zeros(self.S.shape, dtype=self.S.dtype)
            self.S_prime_prime[non_zero_indices] = (2*self.alpha - 1) * self.S[non_zero_indices]**(2*self.alpha-2)
            self.computed_hessian = True
            self.S_inv = np.zeros(self.S.shape, dtype=self.S.dtype)
            self.S_inv[non_zero_indices] = 1 / self.S[non_zero_indices]
        # Compute dP and dD
        l, D1, D2, r = self.theta.shape
        dUtheta = np.tensordot(dU, self.theta, ([2, 3], [1, 2])) # i j [i*] [j*]; l [i] [j] r -> i j l r { D^8 }
        dUtheta = dUtheta.transpose(2, 0, 1, 3).reshape(l*D1, D2*r) # i, j, l, r -> l, i, j, r
        dP = np.conj(self.X.T)@dUtheta@self.Y
        dD = 1.j * np.imag(np.diag(dP)) * self.S_inv / 2
        # Compute dX, dS and dY
        dX = self.X@(self.F * (dP*self.S + self.S[:, np.newaxis]*np.conj(dP.T)) + np.diag(dD)) + (np.eye(l*D1) - self.X@np.conj(self.X.T))@dUtheta@self.Y * self.S_inv
        dS =np.real(np.diag(dP))
        dY = self.Y@(self.F * (self.S[:, np.newaxis]*dP + np.conj(dP.T)*self.S) - np.diag(dD)) + (np.eye(D2*r) - self.Y@np.conj(self.Y.T))@np.conj(dUtheta.T)@self.X * self.S_inv
        # Compute first term of the product rule
        result = -2*self.alpha / self.S2alpha**2 * np.sum(self.S_prime*dS) * self.grad
        # Compute helper tensor necessary for computing the first two of the remaining four terms
        dXSY = (self.X*self.S_prime_prime*dS)@np.conj(self.Y.T)
        dXSY += (dX*self.S_prime)@np.conj(self.Y.T)
        dXSY += (self.X*self.S_prime)@np.conj(dY.T)
        dXSY = dXSY.reshape(l, D1, D2, r)
        # first of four remaining terms
        temp_result = np.tensordot(dXSY, np.conj(self.theta), ([0, 3], [0, 3])) # [l] i j [r]; [l*] k* l* [r*] -> i j k* l*
        # second of four remaining terms
        temp = np.tensordot(np.conj(temp_result), self.U, ([2, 3], [2, 3])) # i* j* [k*] [l*]; i j [k] [l] -> i* j* i j
        temp = np.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i*] [j*] i j; [i] [j] k l -> i j k l
        temp_result -= temp
        # third of four remaining terms
        temp = np.tensordot(np.conj(self.XSprimeYtheta), dU, ([2, 3], [2, 3])) # i* j* [k*] [l*]; i j [k] [l] -> i* j* i j
        temp = np.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i*] [j*] i j; [i] [j] k l -> i j k l
        temp_result -= temp
        # fourth of four remaining terms
        temp = np.tensordot(np.conj(self.XSprimeYtheta), self.U, ([2, 3], [2, 3])) # i* j* [k*] [l*]; i j [k] [l] -> i* j* i j
        temp = np.tensordot(temp, dU, ([0, 1], [0, 1])) # [i*] [j*] i j; [i] [j] k l -> i j k l
        temp_result -= temp
        # Compute final result
        result = self.alpha / (1-self.alpha) * (result + temp_result / self.S2alpha)
        # Project to tangent space and return
        U = self.U.reshape(self.D1*self.D2, -1)
        result = result.reshape(U.shape)
        return (result - 0.5 * U@(U.conj().T@result + result.conj().T@U)).reshape(self.D1, self.D2, self.D1, self.D2)

class RenyiAlphaIterateApproxTRM(RenyiAlphaIterate):
    """
    Class representing a single iterate of the approximate renyi-alpha trust region optimizer.
    Cashes important results such that they do not need to be computed multiple times.
    This is especially useful when multiple hessian vector products must be computed, dropping the complexity
    from O(D^8) to O(D^7) for all calls except for the first call.
    """

    def __init__(self, U, theta, alpha=0.5, chi_max=None, N_iters_svd=None, eps_svd=0, old_iterate=None):
        """
        Initializes new iterate.

        Parameters
        ----------
        See class RenyiAlphaIterate.
        """
        RenyiAlphaIterate.__init__(self, U, theta, alpha, chi_max=chi_max, N_iters_svd=N_iters_svd, eps_svd=eps_svd, old_iterate=old_iterate)
        self.computed_gradient = False
        self.computed_hessian = False

    def evaluate_cost_function(self):
        """
        Computes the approximate value of the renyi alpha entropy 1/(1 - alpha) * log(sum_{i} s_i^(2*alpha)), with s_i the ith singular value
        of U@theta.

        Returns
        -------
        cost : float
            value of the cost function
        """
        # Compute Utheta
        self.Utheta = np.tensordot(self.U, self.theta, ([2, 3], [1, 2])) # i j [i*] [j*]; l [i] [j] r -> i j l r { D^8 }
        self.Utheta = self.Utheta.transpose(2, 0, 1, 3) # i, j, l, r -> l, i, j, r
        # Perform SVD
        l, i, j, r = self.Utheta.shape
        if self.N_iters_svd is None:
            self.X, self.S, self.Y = utility.safe_svd(self.Utheta.reshape(l*i, j*r), full_matrices=False) # { D^9 }
            self.Y = np.conj(self.Y.T)
            if self.chi_max is not None:
                idx = np.argsort(self.S)[::-1][:self.chi_max]
                self.X, self.S, self.Y = self.X[:, idx], self.S[idx], self.Y[:, idx]
        else:
            self.X, self.Y, _, _ = utility.split_matrix_iterate_QR(self.Utheta.reshape(l*i, j*r), self.chi_max, self.N_iters_svd, self.eps_svd, C0=self.Y0, normalize=False) # { N_iters_svd * D^7 }
            self.Y0 = self.Y
            XX, self.S, self.Y = np.linalg.svd(self.Y, full_matrices=False) # { D^5 }
            self.X = self.X@XX
            self.Y = np.conj(self.Y.T)
        # Cache helper variables
        self.S2alpha = np.sum(self.S**(2*self.alpha))
        # Compute and return renyi-alpha entropy
        return 1/(1-self.alpha)*np.log(np.real(self.S2alpha))

    def compute_gradient(self):
        """
        Computes the approximate gradient of the renyi alpha entropy, projected to the tangent space of the iterate.

        Returns
        -------
        grad : np.ndarray of shape (i, j, i*, j*)
            gradient of the cost function
        """
        if not self.computed_gradient:
            self.S_prime = self.S**(2*self.alpha-1)
            self.Xtheta = np.tensordot(self.X.reshape(self.l, self.D1, self.X.shape[-1]), np.conj(self.theta), ([0], [0])) # [l] i chi; [l*] k* l* r* -> i chi k* l* r* { D^8 }
            self.XSprimeYtheta = np.tensordot(self.Xtheta, np.diag(self.S_prime), ([1], [0])) # i [chi] k* l* r*; [chi] chi -> i k* l* r* chi { D^7 }
            self.XSprimeYtheta = np.tensordot(self.XSprimeYtheta, np.conj(self.Y.reshape(self.D2, self.r, self.Y.shape[-1])), ([3, 4], [1, 2])) # i k* l* [r*] [chi]; j [r] [chi] -> i k* l* j  { D^7 }
            self.XSprimeYtheta = self.XSprimeYtheta.transpose(0, 3, 1, 2) # i, k*, l*, j -> i, j, k*, l*
            self.XSprimeYthetaUU = np.tensordot(np.conj(self.XSprimeYtheta), self.U, ([2, 3], [2, 3])) # i' j' [k'] [l']; i j [k'] [l'] -> i' j' i j { D^6 }
            self.XSprimeYthetaUU = np.tensordot(self.XSprimeYthetaUU, self.U, ([0, 1], [0, 1])) # [i'] [j'] i j; [i'] [j'] k l -> i j k l { D^6 }
            self.computed_gradient = True
        return self.alpha / (1 - self.alpha) / self.S2alpha * (self.XSprimeYtheta - self.XSprimeYthetaUU)
    
    def compute_hessian_vector_product(self, dU):
        """
        Computes the approximate riemannian hessian vector product of dU, which is the projection of D(grad f(U))[dU] to the tangent space of U. 
        When this is called for the first time, several intermediate results are cached for the case that this is called again in the future.

        Parameters
        ----------
        dU : np.ndarray of shape (i, j, i*, j*)
            element from the tangent space of U

        Returns
        -------
        hvp : np.ndarray of shape (i, j, i*, j*)
            hessian vector product H@dU
        """
        if not self.computed_hessian:
            # Cache helper results
            self.F = np.zeros((self.k, self.k)) # { D^2 }
            for i in range(self.k):
                for j in range(self.k):
                    if i != j:
                        temp = self.S[j]**2 - self.S[i]**2
                        if temp == 0.0:
                            self.F[i, j] = 0.0
                        else:
                            self.F[i, j] = 1/temp
            self.XthetaY = np.tensordot(self.Xtheta, np.conj(self.Y.reshape(self.D2, self.r, -1)), ([4], [1])) # i chi k* l* [r*]; j* [r*] chi* -> i chi k* l* j* chi* { D^8 }
            self.XthetaY = self.XthetaY.transpose(0, 4, 2, 3, 1, 5) # i, chi, k*, l*, j*, chi* -> i, j, k, l, chi, chi* { D^6 }
            self.thetaY = np.conj(np.tensordot(self.theta, self.Y.reshape(self.D2, self.r, -1), ([3], [1]))) # l* k* l* [r*]; j* [r*] chi* -> l* k* l* j* chi* { D^8 }
            self.thetaY = self.thetaY.transpose(0, 3, 1, 2, 4) # l*, k*, l*, j*, chi* -> l*, j*, k*, l*, chi* { D^6 }
            # Compute modified inverse of S
            self.invS = np.zeros(self.S.size, dtype=self.S.dtype)
            S_nonzero = np.where(self.S != 0.0)
            self.invS[S_nonzero] = 1/self.S[S_nonzero]
            # Compute ImXXT and InYYT
            self.ImXXT = np.eye(self.X.shape[0]) - self.X@np.conj(self.X.T) # { D^7 }
            self.ImXXT = self.ImXXT.reshape(self.l, self.D1, self.l, self.D1) # { D^6 }
            self.InYYT = np.eye(self.Y.shape[0]) - self.Y@np.conj(self.Y.T) # { D^7 }
            self.InYYT = self.InYYT.reshape(self.D2, self.r, self.D2, self.r) # { D^6 }
            # Compute second derivative of S
            self.S_prime_prime = np.zeros(self.S.size, dtype=self.S.dtype) # When detecting a zero in S, we set S_prime_prime to zero.
            self.S_prime_prime[S_nonzero] = (2*self.alpha - 1) * self.S[S_nonzero]**(2*self.alpha-2)
            # self.S_prime_prime = (2*self.alpha - 1) * self.S**(2*self.alpha-2)
            self.computed_hessian = True

        temp = np.tensordot(dU, np.conj(self.thetaY), ([1, 2, 3], [1, 2, 3])) # i [j] [k] [l]; l* [j*] [k*] [l*] chi* -> i l* chi* { D^7 }
        dP = np.tensordot(np.conj(self.X).reshape(self.l, self.D1, -1), temp, ([0, 1], [1, 0])) # [l*] [i*] chi; [i] [l*] chi* -> chi chi* { D^5 } 
        dD = np.diag(1.j*(np.imag(np.diag(dP)) * self.invS / 2))
        temp = np.tensordot(self.ImXXT, temp, ([2, 3], [1, 0])) * self.invS # l i [l] [i]; [i] [l*] chi* -> l i chi* { D^7 }
        temp = temp.reshape(self.X.shape)
        dX = self.X@(self.F * (dP*self.S + self.S[:, np.newaxis]*np.conj(dP.T)) + dD) + temp # { D^4 }
        dS = np.real(np.diag(dP))
        temp = np.tensordot(self.Xtheta, np.conj(dU), ([0, 2, 3], [0, 2, 3])) # [i] chi [k*] [l*] r*; [i] j [k] [l] -> chi r* j { D^7 }
        temp = np.tensordot(self.InYYT, temp, ([2, 3], [2, 1])) * self.invS # j r [j] [r]; chi [r*] [j] -> j r chi { D^7 }
        temp = temp.reshape(self.Y.shape)
        dY = self.Y@(self.F * (self.S[:, np.newaxis]*dP + np.conj(dP.T)*self.S) - dD) + temp # { D^4 }

        # Compute first of five terms
        result_1 = -2*self.alpha / self.S2alpha**2 * np.sum(self.S_prime * dS) * (self.XSprimeYtheta - self.XSprimeYthetaUU) # { D^4 }
        # Compute second of five terms
        temp = np.tensordot(self.XthetaY, np.diag(self.S_prime_prime*dS), ([4, 5], [0, 1])) # i j k l [chi] [chi]; [chi] [chi] -> i j k l { D^6 }
        temp2 = np.tensordot(self.thetaY, np.diag(self.S_prime), ([4], [1])) # l* j* k* l* [chi*]; chi [chi] -> l j k l chi { D^6 }
        temp += np.tensordot(dX.reshape(self.l, self.D1, -1), temp2, ([0, 2], [0, 4])) # [l] i [chi]; [l] j k l [chi] -> i j k l { D^7 }
        temp2 = np.tensordot(self.Xtheta, np.diag(self.S_prime), ([1], [0])) # i [chi] k* l* r*; [chi] chi -> i k l r chi { D^6 }
        temp += np.tensordot(temp2, np.conj(dY.reshape(self.D2, self.r, -1)), ([3, 4], [1, 2])).transpose(0, 3, 1, 2) # i k l [r] [chi]; j [r] [chi] -> i k l j -> i j k l { D^7 }
        result_2 = temp
        # Compute third of five terms
        temp = np.tensordot(np.conj(temp), self.U, ([2, 3], [2, 3])) # i j [k] [l]; i j [k] [l] -> i j i j { D^6 }
        result_2 -= np.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i] [j] i j; [i] [j] k l -> i j k l { D^6 }
        # Compute fourth of five terms
        temp = np.tensordot(np.conj(self.XSprimeYtheta), dU, ([2, 3], [2, 3])) # i j [k] [l]; i j [k] [l] -> i j i j { D^6 }
        result_2 -= np.tensordot(temp, self.U, ([0, 1], [0, 1])) # [i] [j] i j; [i] [j] k l -> i j k l { D^6 }
        # Compute fifth of five terms
        temp = np.tensordot(np.conj(self.XSprimeYtheta), self.U, ([2, 3], [2, 3])) # i j [k] [l]; i j [k] [l] -> i j i j { D^6 }
        result_2 -= np.tensordot(temp, dU, ([0, 1], [0, 1])) # [i] [j] i j; [i] [j] k l -> i j k l { D^6 }

        # Compute final result
        result = (self.alpha / (1 - self.alpha) * (result_1 + result_2 /self.S2alpha)).reshape(self.D1*self.D2, -1)
        # Project to tangent space and return
        U = self.U.reshape(result.shape)
        return (result - 0.5 * U@(U.conj().T@result + result.conj().T@U)).reshape(self.D1, self.D2, self.D1, self.D2)