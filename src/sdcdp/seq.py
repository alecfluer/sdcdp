import numbers
import random
import warnings

import numpy as np
import scipy

def sample_degree_sequence(mean_degree_sequence):
    """
    Samples a degree sequence (of nonnegative integers) from a mean degree sequence (of nonnegative values)
    for an undirected network, and ensures the sum of the degree sequence is even. The sampled degrees are
    drawn independently from a Poisson distribution using the provided mean degrees.

    See docs/seq.md for documentation and full mathematical details.

    Parameters
    ----------
    mean_degree_sequence : array-like
        The mean degree sequence with shape (N,). Entry i corresponds to the mean degree for node i.

    Returns
    -------
    degree_sequence : array-like
        The sampled degree sequence with shape (N,). Entry i corresponds to the sampled degree for node i.
    """

    mean_degree_sequence = np.array(mean_degree_sequence)
    if np.any(mean_degree_sequence < 0):
        raise ValueError("Invalid Input: Each entry in 'mean_degree_sequence' must be nonnegative.")
    degree_sequence = np.array([scipy.stats.poisson.rvs(mu=mean) for mean in mean_degree_sequence])
    if np.sum(degree_sequence) % 2 != 0:
        degree_sequence[random.choices(population=np.arange(len(degree_sequence)), k=1)[0]] += 1

    return degree_sequence

def sample_correlated_mean_degree_sequences(means, variances, size, omega=None):
    """
    Samples correlated mean degree sequences (of nonnegative values) for an undirected multiplex network.
    The sampled mean degrees are drawn from a multivariate lognormal distribution using specified means,
    variances, and a correlation matrix.

    See docs/seq.md for documentation and full mathematical details.

    Parameters
    ----------
    means : array-like
        The means of the degree sequences as an array with shape (L,).

    variances : array-like
        The variances of the degree sequences as an array with shape (L,).

    size : int
        The number of nodes in the network.

    omega : array-like or None, optional
        A symmetric correlation matrix with shape (L, L). If None, the identity matrix is used.
        Default is None.

    Returns
    -------
    correlated_sequences : array-like
        The sampled correlated mean degree sequences with shape (L, N).
    """

    if not isinstance(means, (list, tuple, np.ndarray)):
        raise TypeError("Invalid Input: 'means' must be a list, tuple, or NumPy array.")
    if not isinstance(variances, (list, tuple, np.ndarray)):
        raise TypeError("Invalid Input: 'variances' must be a list, tuple, or NumPy array.")
    if not isinstance(omega, (list, np.ndarray, type(None))):
        raise TypeError("Invalid Input: 'omega' must be a list-of-lists, NumPy array, or None.")
    
    means = np.array(means)
    variances = np.array(variances)
    layers = len(means)

    if omega is None and len(means) != len(variances):
        raise ValueError("Invalid Input: 'means' and 'variances' must have shape (L,).")
    
    omega = np.eye(layers) if omega is None else np.array(omega)

    if omega is not None and (len(means) != len(variances) or omega.shape != (layers, layers)):
        raise ValueError("Invalid Input: 'means' and 'variances' must have shape (L,) and 'omega' must have shape (L, L).")
    
    if not np.issubdtype(means.dtype, np.number) or np.any(means <= 0):
        raise ValueError("Invalid Input: each entry in 'means' must be numeric and greater than 0.")
    if not np.issubdtype(variances.dtype, np.number) or np.any(variances <= means):
        raise ValueError("Invalid Input: each entry in 'variances' must be numeric and greater than corresponding entry in 'means'.")
    if not np.issubdtype(omega.dtype, np.number) or np.any(np.abs(omega) > 1):
        raise ValueError("Invalid Input: each entry in 'omega' must be numeric and bounded between -1 and 1.")
    if np.any(omega.diagonal() != 1):
            raise ValueError("Invalid Input: each diagonal entry in 'omega' must be numeric and equal to 1.")
    if not np.array_equal(omega, omega.T):
        raise ValueError("Invalid Input: 'omega' must be symmetric.")
    
    if not isinstance(size, numbers.Number) or size < 0 or size % 1 != 0:
        raise ValueError("Invalid Input: 'size' must be a non-negative integer.")

    taus = []
    thetas = []
    for i in range(layers):
        mean = means[i]
        var = variances[i]
        tau = np.sqrt(np.log(1 + ((var - mean) / (mean**2))))
        theta = np.log(mean / np.sqrt(1 + ((var - mean) / (mean**2))))
        taus.append(tau)
        thetas.append(theta)
    
    taus = np.diag(taus)
    thetas = np.array(thetas)
    sigma = np.matmul(taus, np.matmul(omega, taus))

    evals, evecs = np.linalg.eig(sigma)
    if np.any(np.real(evals) < 0):
        evals[np.real(evals) < 10e-14] = 10e-14
        cov = np.array(sigma)
        sigma = np.matmul(evecs, np.matmul(np.diag(evals), evecs.T))
        warnings.warn(f"Computation Error. Computed covariance matrix Sigma not a positive semidefinite matrix. "
                      f"Took covariance matrix to be nearest positive semidefinite matrix in the Frobenius norm. "
                      f"Frobenius norm of the difference: {np.linalg.norm(cov - sigma)}. "
                      f"See documentation for details.", RuntimeWarning)
        
    samples = scipy.stats.multivariate_normal.rvs(mean=thetas, cov=sigma, size=size)
    correlated_sequences = np.exp(samples).T

    return correlated_sequences

def compute_correlation_from_pearson(means, variances, pearson=None, tol=1):
    """
    Constructs a correlation matrix suitable for sampling degree sequences from Pearson correlation
    coefficients. The resulting correlation matrix is used to construct a lognormal covariance matrix
    for sampling correlated mean degree sequences.

    See docs/seq.md for documentation and full mathematical details.

    Parameters
    ----------
    means : array-like
        The means of the degree sequences as an array with shape (L,).

    variances : array-like
        The variances of the degree sequences as an array with shape (L,).

    pearson : array-like or None, optional
        Pearson correlation coefficients as a symmetric matrix with shape (L, L). If None, the identity
        matrix is used. Default is None.

    tol : float or 1, optional
        Tolerance for numerical stability. Default is 1.

    Returns
    -------
    omega : array-like
        A symmetric correlation matrix with shape (L, L).
    """

    if not isinstance(means, (list, tuple, np.ndarray)):
        raise TypeError("Invalid Input: 'means' must be a list, tuple, or NumPy array.")
    if not isinstance(variances, (list, tuple, np.ndarray)):
        raise TypeError("Invalid Input: 'variances' must be a list, tuple, or NumPy array.")
    if not isinstance(pearson, (list, np.ndarray, type(None))):
        raise TypeError("Invalid Input: 'pearson' must be a list-of-lists, NumPy array, or None.")
    
    means = np.array(means)
    variances = np.array(variances)
    layers = len(means)

    if pearson is None and len(means) != len(variances):
        raise ValueError("Invalid Input: 'means' and 'variances' must have shape (L,).")
    
    pearson = np.eye(layers) if pearson is None else np.array(pearson)

    if pearson is not None and (len(means) != len(variances) or pearson.shape != (layers, layers)):
        raise ValueError("Invalid Input: 'means' and 'variances' must have shape (L,) and 'pearson' must have shape (L, L).")
    
    if not np.issubdtype(means.dtype, np.number) or np.any(means <= 0):
        raise ValueError("Invalid Input: each entry in 'means' must be numeric and greater than 0.")
    if not np.issubdtype(variances.dtype, np.number) or np.any(variances <= means):
        raise ValueError("Invalid Input: each entry in 'variances' must be numeric and greater than corresponding entry in 'means'.")
    if not np.issubdtype(pearson.dtype, np.number) or np.any(np.abs(pearson) > 1):
        raise ValueError("Invalid Input: each entry in 'pearson' must be numeric and bounded between -1 and 1.")
    if np.any(pearson.diagonal() != 1):
            raise ValueError("Invalid Input: each diagonal entry in 'pearson' must be numeric and equal to 1.")
    if not np.array_equal(pearson, pearson.T):
        raise ValueError("Invalid Input: 'pearson' must be symmetric.")
    
    if not isinstance(tol, numbers.Number) or tol < 1:
        raise ValueError("Invalid Input: 'tol' must be numeric and greater than or equal to 1.")

    taus = []
    for i in range(layers):
        mean = means[i]
        var = variances[i]
        tau = np.sqrt(np.log(1 + ((var - mean) / (mean**2))))
        taus.append(tau)
    
    taus = np.array(taus)
    omega = np.zeros((layers, layers))

    for i in range(layers):
        for j in range(i):
            pear_r = pearson[i, j]
            mean_i = means[i]
            mean_j = means[j]
            var_i = variances[i]
            var_j = variances[j]
            tau_i = taus[i]
            tau_j = taus[j]

            if -1 <= pear_r < -((mean_i * mean_j) / np.sqrt(var_i * var_j)):
                raise ValueError(
                    f"Computation Error. Provided Pearson correlation coefficient between layers "
                    f"{i} and {j} incompatible with provided means and variances. Ensure 'pearson', "
                    f"'means', and 'variances' are compatible. See documentation for details."
                )

            alpha = np.log((pear_r * (np.sqrt(var_i * var_j) / (mean_i * mean_j))) + 1) / (tau_i * tau_j)

            if np.abs(alpha) > 1:
                if np.abs(alpha) <= tol:
                    warnings.warn(f"Computation Error. Computed correlation coefficient alpha between layers "
                                  f"{i} and {j} exceeds {np.sign(alpha)}. Clipped to {np.sign(alpha)} based "
                                  f"on 'tol={tol}'. See documentation for details.", RuntimeWarning)
                    alpha = np.sign(alpha)
                else:
                    raise ValueError(
                        f"Computation Error. Computed correlation coefficient alpha between layers "
                        f"{i} and {j} exceeds {np.sign(alpha)}. Set 'tol' > 1 to clip all alpha "
                        f"within 'tol'. See documentation for details."
                    )
            
            omega[i, j] = alpha
    
    omega = omega + omega.T
    np.fill_diagonal(omega, 1)

    return omega