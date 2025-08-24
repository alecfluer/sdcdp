# **SDA**

## **Overview**

The **Social Distance Attachment (SDA)** method generates undirected connection probabilities between nodes based on latent social space features. Each social feature encodes a node-level attribute — such as geographic location or demographic identity — of arbitrary dimension, and is paired with a set of node locations and a distance function that governs pairwise separation.

SDA was originally proposed by [Boguñá et al. (2004)](#references) for single-layer networks. This extension was proposed by [Fluer et al. (2025)](#references) and generalizes SDA to multiplex networks. Each network layer defines its own connection probabilities over a shared set of social features. While node locations and pairwise distances for each social feature remain fixed across network layers, the influence of each social feature can vary between network layers.

This extension is implemented in the `MultiplexSDA` class, which manages the assignment of network layers, social features, and (network layer, social feature) pair parameters. It computes and returns connection probabilities as a dictionary mapping each network layer to a symmetric probability matrix.

## **Mathematical Formulation**

Let $\mathcal{G}=(\mathcal{N},\mathcal{E},\mathcal{L})$ be a multiplex network with $|\mathcal{N}|=N$ nodes and $|\mathcal{L}|=L$ network layers. Let $\mathcal{F}$ be a real-valued $\dim({\mathcal{F}})$-dimensional social space. Let $\vec x_i=(x_{i,1},\dots,x_{i,\dim(\mathcal{F})})\in\mathcal{F}$ be the position of node $i$ in this space, where each coordinate $x_{i,f}$ denotes the location of node $i$ in a distinctive social feature, which may itself be multidimensional.

Given a network layer indexed by $l$, the connection probability between nodes $i$ and $j$ is computed as a convex combination of social-feature-restricted attachment probabilities:

$$
\underbrace{p_{ij}^l=\sum_{f=1}^{\dim(\mathcal{F})}\omega_f^l \cdot p_{ij,f}^l}_{\text{connection probability}}
\quad
,
\quad
\underbrace{\vphantom{\sum_{f=1}^{\dim(\mathcal{F})}}p_{ij,f}^l=\frac{1}{1+[(b_f^l)^{-1}\delta_f(x_{i,f},x_{j,f})]^{h_f^l}}}_{\text{social-feature-restricted attachment probability}}
$$

Given a social feature indexed by $f$:

- $x_{i,f}$ is the location of node $i$

- $\delta_f(\cdot,\cdot)$ is a distance function that quantifies separation between nodes

- $b_f^l>0$ is a characteristic distance or length scale at which $p_{ij,f}^l=0.5$

- $h_f^l \geq 1$ is a homophily parameter that describes the tendency of similar nodes to form connections

- $\omega_f^l\geq 0$ is a weight coefficient that describes the contribution of the social feature, where $\sum_f\omega_f^l=1$

## **MultiplexSDA Attributes**

| Attribute | | |
|---|---|---|
| [**networks_dataframe**](#networks_dataframe) | ***pd.DataFrame*** | Stores characteristic distance, homophily, and weight for (network layer, social feature) pairs. |
| [**features_dataframe**](#features_dataframe) | ***pd.DataFrame*** | Stores node locations and distance function for social features. |
| **dataframe** | ***pd.DataFrame*** | Returns a merged view of [**networks_dataframe**](#networks_dataframe) and [**features_dataframe**](#features_dataframe). |
| **networks** | ***np.ndarray*** | Returns an array of unique network layer names. |
| **features** | ***np.ndarray*** | Returns an array of unique social feature names. |
| **number_of_networks** | ***int or None*** | Returns the number of unique network layers. |
| **number_of_features** | ***int or None*** | Returns the number of unique social features. |
| **number_of_nodes** | ***int or None*** | Returns the number of nodes inferred from the first assigned social feature. |
| **shape** | ***tuple*** | Returns `(number_of_networks, number_of_features, number_of_nodes)`. |

---

### [**networks_dataframe**](#networks_dataframe)

`MultiplexSDA.networks_dataframe`

Each row corresponds to a (network layer, social feature) pair, with the following columns:

| Column        |             |                                      |
|---------------|-------------|--------------------------------------|
| **network**   | ***str***   | Name of the network layer $l$.       |
| **feature**   | ***str***   | Name of the social feature $f$.      |
| **char_dist** | ***float*** | The characteristic distance $b_f^l$. |
| **homophily** | ***float*** | The homophily parameter $h_f^l$.     |
| **weight**    | ***float*** | The weight coefficient $\omega_f^l$. |

---

### [**features_dataframe**](#features_dataframe)

`MultiplexSDA.features_dataframe`

Each row corresponds to a social feature, with the following columns:

| Column        |                  |                                                                                                  |
|---------------|------------------|--------------------------------------------------------------------------------------------------|
| **feature**   | ***str***        | Name of the social feature $f$.                                                                  |
| **locations** | ***array-like*** | The node locations $(x_{i,f})_{i=1}^N$ as an array with shape `(N,)` or `(N, d)`.                |
| **dist_func** | ***callable***   | The distance function $\delta_f(\cdot,\cdot)$ as a callable function on $x_{i,f}$ and $x_{j,f}$. |

## **MultiplexSDA Methods**

| Method | |
|--------|-|
| [**add_networks_from**](#add_networks_from) | Adds one or more network layers to the model. |
| [**add_features_from**](#add_features_from) | Adds one or more social features to the model. |
| [**remove_networks_from**](#remove_networks_from) | Removes one or more network layers from the model. |
| [**remove_features_from**](#remove_features_from) | Removes one or more social features from the model. |
| [**assign_network_params**](#assign_network_params) | Assigns characteristic distance, homophily, and weight to a specified (network layer, social feature) pair. |
| [**assign_feature_params**](#assign_feature_params) | Assigns node locations and distance function to a specified social feature. |
| [**clear_network_params_from**](#clear_network_params_from) | Clears characteristic distance, homophily, and weight from specified (network layer, social feature) pairs. |
| [**clear_feature_params_from**](#clear_feature_params_from) | Clears node locations and distance function from specified social features. |
| [**compute_dist_matrices**](#compute_dist_matrices) | Computes pairwise distances between nodes using node locations and distance function for each specified social feature. |
| [**compute_prob_matrices**](#compute_prob_matrices) | Computes SDA connection probabilities between nodes using available class data for each specified network layer. |

---

### [**add_networks_from**](#add_networks_from)

`MultiplexSDA.add_networks_from(networks)`

Adds one or more network layers to the model.

Each new network layer is paired with all existing social features. If no social features exist, placeholder rows are added with `np.nan` values.

| Parameters   |                         |                                     |
|--------------|-------------------------|-------------------------------------|
| **networks** | ***array-like of str*** | Names of the network layers to add. |

---

### [**add_features_from**](#add_features_from)

`MultiplexSDA.add_features_from(features)`

Adds one or more social features to the model.

Each new social feature is paired with all existing network layers. If no network layers exist, placeholder rows are added with `np.nan` values.

| Parameters   |                         |                                      |
|--------------|-------------------------|--------------------------------------|
| **features** | ***array-like of str*** | Names of the social features to add. |

---

### [**remove_networks_from**](#remove_networks_from)

`MultiplexSDA.remove_networks_from(networks)`

Removes one or more network layers from the model.

Each removed network layer is deleted from all existing social features.

| Parameters   |                         |                                        |
|--------------|-------------------------|----------------------------------------|
| **networks** | ***array-like of str*** | Names of the network layers to remove. |

---

### [**remove_features_from**](#remove_features_from)

`MultiplexSDA.remove_features_from(features)`

Removes one or more social features from the model.

Each removed social feature is deleted from all existing network layers.

| Parameters   |                         |                                         |
|--------------|-------------------------|-----------------------------------------|
| **features** | ***array-like of str*** | Names of the social features to remove. |

---

### [**assign_network_params**](#assign_network_params)

`MultiplexSDA.assign_network_params(network, feature, char_dist, homophily, weight, create=False)`

Assigns characteristic distance, homophily, and weight to a specified (network layer, social feature) pair.

| Parameters    |                      |                                                                                                      |
|---------------|----------------------|------------------------------------------------------------------------------------------------------|
| **network**   | ***str***            | Name of the network layer for parameter assignment.                                                  |
| **feature**   | ***str***            | Name of the social feature for parameter assignment.                                                 |
| **char_dist** | ***float***          | The characteristic distance. Must be greater than 0.                                                 |
| **homophily** | ***float***          | The homophily parameter. Must be greater than or equal to 1.                                         |
| **weight**    | ***float***          | The weight coefficient. Must be greater than or equal to 0.                                          |
| **create**    | ***bool, optional*** | If ***True***, adds network layer or social feature to the model if missing. Default is ***False***. |

| Notes | |
|-------|-|
|**1.**| The model internally normalizes weights across all assigned (network layer, social feature) pairs per network layer. |

---

### [**assign_feature_params**](#assign_feature_params)

`MultiplexSDA.assign_feature_params(feature, locations, dist_func, create=False)`

Assigns node locations and distance function to a specified social feature.

| Parameters    |                      |                                                                                     |
|---------------|----------------------|-------------------------------------------------------------------------------------|
| **feature**   | ***str***            | Name of the social feature for parameter assignment.                                |
| **locations** | ***array-like***     | The node locations. Can be multidimensional with shape `(N,)` or `(N, d)`.          |
| **dist_func** | ***callable***       | The distance function. Must be compatible with the node locations.                  |
| **create**    | ***bool, optional*** | If ***True***, adds social feature to the model if missing. Default is ***False***. |

| Notes | |
|-------|-|
| **1.** | The number of nodes in the model is inferred from the node locations of the first assigned social feature. |
| **2.** | The model enforces the same number of nodes across all assigned social features. If an inconsistency is detected, [**assign_feature_params**](#assign_feature_params) will fail. If node locations must be reset, use [**clear_feature_params_from**](#clear_feature_params_from) to clear previously assigned social features. |

---

### [**clear_network_params_from**](#clear_network_params_from)

`MultiplexSDA.clear_network_params_from(networks=None, features=None)`

Clears characteristic distance, homophily, and weight from specified (network layer, social feature) pairs.

The method applies clearing across the Cartesian product of **networks** and **features**. If either is set to ***None***, all network layers or social features are selected accordingly.

| Parameters | | |
|------------|-|-|
| **networks** | ***array-like of str or None, optional*** | Names of the network layers to clear. If ***None***, all network layers are selected. Default is ***None***. |
| **features** | ***array-like of str or None, optional*** | Names of the social features to clear. If ***None***, all social features are selected. Default is ***None***. |

---

### [**clear_feature_params_from**](#clear_feature_params_from)

`MultiplexSDA.clear_feature_params_from(features=None)`

Clears node locations and distance function from specified social features.

| Parameters | | |
|------------|-|-|
| **features** | ***array-like of str or None, optional*** | Names of the social features to clear. If ***None***, all social features are selected. Default is ***None***. |

---

### [**compute_dist_matrices**](#compute_dist_matrices)

`MultiplexSDA.compute_dist_matrices(features=None)`

Computes pairwise distances between nodes using node locations and distance function for each specified social feature.

| Parameters | | |
|------------|-|-|
| **features** | ***array-like of str or None, optional*** | Names of the social features to compute distances for. If ***None***, all social features are selected. Default is ***None***. |

| Returns | | |
|---------|-|-|
| **dist_matrices** | ***dict[str, np.ndarray]*** | Dictionary mapping each social feature name to a symmetric pairwise distance matrix with shape `(N, N)`. |

| Notes | |
|-------|-|
| **1.** | The model enforces the same number of nodes across all assigned social features. Therefore all distance matrices have shape `(N, N)`. |
| **2.** | If a social feature is missing or has unassigned data, the returned distance matrix will be filled with `np.nan`. |
| **3.** | In each distance matrix, entry $(i, j)$ represents the distance between node $i$ and node $j$. |
| **4.** | A warning is issued if a distance matrix is not symmetric. But the computation still completes. |

---

### [**compute_prob_matrices**](#compute_prob_matrices)

`MultiplexSDA.compute_prob_matrices(networks=None)`

Computes SDA connection probabilities between nodes using available class data for each specified network layer.

| Parameters | | |
|------------|-|-|
| **networks** | ***array-like of str or None, optional*** | Names of the network layers to compute probabilities for. If ***None***, all network layers are selected. Default is ***None***. |

| Returns | | |
|---------|-|-|
| **prob_matrices** | ***dict[str, np.ndarray]*** | Dictionary mapping each network layer name to a symmetric probability matrix with shape `(N, N)`. |

| Notes | |
|-------|-|
| **1.** | The model enforces the same number of nodes across all assigned social features. Therefore all probability matrices have shape `(N, N)`. |
| **2.** | If a network layer is missing or has unassigned data, the returned probability matrix will be filled with `np.nan`. |
| **3.** | If a (network layer, social feature) pair is incomplete or has unassigned data, it is excluded from the computation. |
| **4.** | In each probability matrix, entry $(i, j)$ represents the SDA connection probability between node $i$ and node $j$. |
| **5.** | A warning is issued if a probability matrix is not symmetric. But the computation still completes. |

## **Examples**

```python
import numpy as np
import sdcdp

# Step 1: Initialize the model

model = sdcdp.sda.MultiplexSDA()

# Step 2: Add network layers and social features

model.add_networks_from(networks=["friendships", "collaborations"])
model.add_features_from(features=["age", "hobbies", "geographic"])

# Step 3: Assign social feature parameters

network_size = 100
age_locations = np.random.randint(21, 50, size=network_size)
hobbies_locations = np.random.randint(0, 8, size=network_size)
geographic_locations = np.random.uniform(0, 1, size=(network_size, 2))

def absolute(x, y): return abs(x - y)
def discrete(x, y): return 1 if x == y else 0
def euclidean(x, y): return np.linalg.norm(x - y)

model.assign_feature_params(feature="age", locations=age_locations, dist_func=absolute)
model.assign_feature_params(feature="hobbies", locations=hobbies_locations, dist_func=discrete)
model.assign_feature_params(feature="geographic", locations=geographic_locations, dist_func=euclidean)

# Step 4: Assign (network layer, social feature) pair parameters

model.assign_network_params(network="friendships", feature="age", char_dist=5, homophily=8, weight=0.5)
model.assign_network_params(network="friendships", feature="hobbies", char_dist=1, homophily=4, weight=0.25)
model.assign_network_params(network="friendships", feature="geographic", char_dist=0.5, homophily=4, weight=0.25)

model.assign_network_params(network="collaborations", feature="age", char_dist=20, homophily=6, weight=0.3)
model.assign_network_params(network="collaborations", feature="hobbies", char_dist=1, homophily=4, weight=0)
model.assign_network_params(network="collaborations", feature="geographic", char_dist=1, homophily=2, weight=0.7)

# Step 5: Compute connection probabilities

prob_matrices = model.compute_prob_matrices()

# Step 6: Inspect results

print(f"Connection probabilities (friendships layer):\n\n{prob_matrices['friendships']}\n")
print(f"Connection probabilities (collaborations layer):\n\n{prob_matrices['collaborations']}\n")
```

## **References**

Marián Boguñá, Romualdo Pastor-Satorras, Albert Díaz-Guilera, and Alex Arenas.  
Models of social networks based on social distance attachment. *Physical Review E*, 70(5):056122, November 2004.

Alec Fluer, Ian Laga, Logan Graham, Ellen Almirol, Makenna Meyer, and Breschine Cummins.  
From survey data to social multiplex models: Incorporating interlayer correlation from multiple data sources. In preparation.