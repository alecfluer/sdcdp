# **SEQ**

## **Overview**

This module samples correlated degree sequences for multiplex networks. It is based on the method proposed by [Fluer et al. (2025)](#references). Given a multiplex network where the mean and variance of the degree sequences are known for each network layer, and where the desired correlation — such as that derived from the Pearson correlation coefficient — is known between network layers, this module samples node degrees from a Poisson distribution whose mean node degrees are sampled from a multivariate lognormal distribution. This approach is intended to enforce the desired statistics and scale to any network size.

These computations are implemented in the `sample_degree_sequence`, `sample_correlated_mean_degree_sequences`, and `compute_correlation_from_pearson` functions.

## **Mathematical Formulation**

Let $\mathcal{G}=(\mathcal{N},\mathcal{E},\mathcal{L})$ be a multiplex network with $|\mathcal{N}|=N$ nodes and $|\mathcal{L}|=L$ network layers.

Let $\vec d_l=(d_{1l},\dots,d_{Nl})$, where each coordinate $d_{il}$ denotes the degree of node $i$ in network layer $l$. Let $\vec \mu_i=(\mu_{i1},\dots,\mu_{iL})$, where each coordinate $\mu_{il}$ denotes the mean degree of node $i$ in network layer $l$. These are sampled according to:

$$
d_{il} \mid \mu_{il} \sim \text{Poisson}(\mu_{il})
$$

$$
\vec \mu_i \sim \text{Multivariate-Lognormal}(\vec \theta, \Sigma) = \exp(\text{MVN}(\vec \theta, \Sigma))
$$

Let $\gamma_l$ and $\sigma^2_l$ be the desired mean and variance of the degree sequence in network layer $l$, respectively, where $0<\gamma_l < \sigma^2_l$. Let $\rho_{kl}=\rho_{lk}$ be the desired Pearson correlation coefficient between the degree sequences in network layers $k$ and $l$. The multivariate lognormal distribution parameters are computed as:

$$
\theta_l = \ln\left(\frac{\gamma_l}{\sqrt{1 + \frac{\sigma^2_l - \gamma_l}{\gamma_l^2}}} \right)
$$

$$
\Sigma = \text{diag}(\vec \tau) \;\Omega \;\text{diag}(\vec \tau)
\quad , \quad
\tau^2_l = \ln\left(1 + \frac{\sigma^2_l - \gamma_l}{\gamma_l^2}\right)
$$

$$
\Omega_{kl} = \Omega_{lk} = \begin{cases} 1, & k = l \\ \alpha_{kl}, & k \neq l \end{cases}
\quad , \quad
\alpha_{kl} = \frac{1}{\tau_k \tau_l} \ln\left(\rho_{kl} \frac{\sigma_k \sigma_l}{\gamma_k \gamma_l} + 1\right)
$$

## **Functions**

### [**sample_degree_sequence**](#sample_degree_sequence)

`sample_degree_sequence(mean_degree_sequence)`

Samples a degree sequence (of nonnegative integers) from a mean degree sequence (of nonnegative values) for an undirected network, and ensures the sum of the degree sequence is even. The sampled degrees are drawn independently from a Poisson distribution using the provided mean degrees.

| Parameters | | |
|------------|-|-|
| **mean_degree_sequence** | ***array-like*** | The mean degree sequence with shape `(N,)`. Entry $i$ corresponds to the mean degree for node $i$. |

| Returns | | |
|---------|-|-|
| **degree_sequence** | ***array-like*** | The sampled degree sequence with shape `(N,)`. Entry $i$ corresponds to the sampled degree for node $i$. |

---

### [**sample_correlated_mean_degree_sequences**](#sample_correlated_mean_degree_sequences)

`sample_correlated_mean_degree_sequences(means, variances, size, omega=None)`

Samples correlated mean degree sequences (of nonnegative values) for an undirected multiplex network. The sampled mean degrees are drawn from a multivariate lognormal distribution using specified means, variances, and a correlation matrix.

| Parameters | | |
|------------|-|-|
| **means** | ***array-like*** | The means of the degree sequences as an array with shape `(L,)`. |
| **variances** | ***array-like*** | The variances of the degree sequences as an array with shape `(L,)`. |
| **size** | ***int*** | The number of nodes in the network. |
| **omega** | ***array-like or None, optional*** | A symmetric correlation matrix with shape `(L, L)`. If ***None***, the identity matrix is used. Default is ***None***. |

| Returns | | |
|---------|-|-|
| **correlated_sequences** | ***array-like*** | The sampled correlated mean degree sequences with shape `(L, N)`. |

| Notes | |
|-------|-|
|**1.**| Parameters are order dependent. In **means** and **variances**, entry $l$ corresponds to the mean $\gamma_l$ and variance $\sigma^2_l$ of the degree sequence in network layer $l$, respectively. In **omega**, entry $(k, l)$ corresponds to the correlation $\alpha_{kl}$ between the degree sequences in network layers $k$ and $l$. |
|**2.**| In **correlated_sequences**, entry $(l, i)$ corresponds to the sampled mean degree for node $i$ in network layer $l$. |
|**3.**| The means, variances, and correlations are used to construct a covariance matrix from which to sample the correlated mean degree sequences. This covariance matrix must be a positive semidefinite matrix. If it is not, the covariance matrix is taken to be the nearest positive semidefinite matrix in the Frobenius norm using the method of [Cheng et al. (1998)](#references). |

---

### [**compute_correlation_from_pearson**](#compute_correlation_from_pearson)

`compute_correlation_from_pearson(means, variances, pearson=None, tol=1)`

Constructs a correlation matrix suitable for sampling degree sequences from Pearson correlation coefficients. The resulting correlation matrix is used to construct a lognormal covariance matrix for sampling correlated mean degree sequences.

| Parameters | | |
|------------|-|-|
| **means** | ***array-like*** | The means of the degree sequences as an array with shape `(L,)`. |
| **variances** | ***array-like*** | The variances of the degree sequences as an array with shape `(L,)`. |
| **pearson** | ***array-like or None, optional*** | Pearson correlation coefficients as a symmetric matrix with shape `(L, L)`. If ***None***, the identity matrix is used. Default is ***None***. |
| **tol** | ***float or 1, optional*** | Tolerance for numerical stability. Default is ***1***. |

| Returns | | |
|---------|-|-|
| **omega** | ***array-like*** | A symmetric correlation matrix with shape `(L, L)`. |

| Notes | |
|-------|-|
|**1.**| Parameters are order dependent. In **means** and **variances**, entry $l$ corresponds to the mean $\gamma_l$ and variance $\sigma^2_l$ of the degree sequence in network layer $l$, respectively. In **pearson**, entry $(k, l)$ corresponds to the Pearson correlation coefficient $\rho_{kl}$ between the degree sequences in network layers $k$ and $l$. |
|**2.**| In **omega**, entry $(k, l)$ corresponds to the computed correlation $\alpha_{kl}$ between the degree sequences in network layers $k$ and $l$. |
|**3.**| Each diagonal entry $\alpha_{ll}$ in **omega** is set equal to 1. |
|**4.**| If any computed entry $\alpha_{kl}$ in **omega** exceeds ±1 but is within ±**tol**, it is clipped to ±1 respectively. |
|**5.**| If $-1 \leq \rho_{kl} < -\frac{\gamma_k\gamma_l}{\sigma_k\sigma_l}$, the estimation fails. |

## **Examples**

```python
import numpy as np
import sdcdp

# Step 1: Specify network size and network layer statistics

size = 100
means = [10, 20]
variances = [40, 80]

# Step 2: Specify Pearson correlation coefficient matrix

pearson = np.array([
    [1.0, 0.6],
    [0.6, 1.0]
])

# Step 3: Compute transformed correlation matrix

omega = sdcdp.seq.compute_correlation_from_pearson(means=means, variances=variances, pearson=pearson, tol=1)

# Step 4: Sample correlated mean degree sequences

mean_degree_sequences = sdcdp.seq.sample_correlated_mean_degree_sequences(means=means, variances=variances, size=size, omega=omega)

# Step 5: Sample integer-valued degree sequences

degree_sequences = [sdcdp.seq.sample_degree_sequence(mean_deg_seq) for mean_deg_seq in mean_degree_sequences]

# Step 6: Inspect results

print("Layer 1 degree sequence:", degree_sequences[0])
print("Layer 2 degree sequence:", degree_sequences[1])
```

## **References**

Sheung Hun Cheng and Nicholas J. Higham.  
A modified cholesky algorithm based on a symmetric indefinite factorization. *SIAM Journal on Matrix Analysis and Applications*, 19(4):1097-1110, January 1998.

Alec Fluer, Ian Laga, Logan Graham, Ellen Almirol, Makenna Meyer, and Breschine Cummins.  
From survey data to social multiplex models: Incorporating interlayer correlation from multiple data sources. In preparation.