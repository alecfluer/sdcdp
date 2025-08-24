# **SDCDP**

## **Overview**

The **Social Distance Configuration model with Degree Preservation (SDC-DP)** is a modified version of the configuration model for sampling networks. It was proposed by [Fluer, et al. (2025)](#references). It samples networks where node degrees preserve a given degree sequence and edge formation is subject to edge formation probabilities. It biases sampling toward simple networks — those with no or few self-loops and multiedges — and supports both undirected and directed networks.

Similar to the configuration model, SDC-DP assigns to each node the number of stubs indicated by its degree, and then assigns edges between nodes iteratively until every stub is connected. However, uniform random stub selection in the configuration model is replaced by non-uniform random node selection in SDC-DP. It selects the first node with probability proportional to its remaining degree, and selects the second node with probability conditioned on the first node and the given edge formation probabilities. It defers assigning self-loops and multiedges until a maximal simple network has been sampled.

The algorithm is implemented in the `sdcdp_model` function.

## **Mathematical Formulation**

### **Undirected Network**

Let $\mathcal{G}=(\mathcal{N},\mathcal{E})$ be a network with $|\mathcal{N}|=N$ nodes. Let $P=(p_{ij})$ be a symmetric probability matrix, where each entry $p_{ij}$ denotes the probability of an edge between node $i$ and node $j$. Let $\vec d = (d_1,\dots,d_N)$ be the degree sequence of the network, where each coordinate $d_i$ denotes the degree of node $i$.

Edges are assigned iteratively by a two-stage scheme. The first node $i$ is selected with probability proportional to its remaining degree:

$$
p(i) = \frac{d_i}{\sum_{k=1}^N d_k}
$$

The second node $j$ is selected with probability conditioned on node $i$:

$$
p(j)=\frac{p_{ij}}{\sum_{k=1}^N p_{ik}}
$$

Once the edge $(i,j)\in\mathcal{E}$ is assigned, the degrees $d_i$ and $d_j$ are decremented, the probability $p_{ij}=p_{ji}=0$ is updated, and the edge is masked from ocurring as a multiedge.

If it is found that any $d_i=0$, the probabilities are updated to $p_{ij}=p_{ji}=0$ for all $j$. If the probability matrix $P=0$ but the sum of the remaining degrees $\sum_{k=1}^N d_k \neq 0$, the algorithm defaults to the undirected configuration model to randomly assign self-loops and multiedges.

### **Directed Network**

Let $\mathcal{G}=(\mathcal{N},\mathcal{E})$ be a network with $|\mathcal{N}|=N$ nodes. Let $P=(p_{ij})$ be a not necessarily symmetric probability matrix, where each entry $p_{ij}$ denotes the probability of an edge from the tail node $i$ to the head node $j$. Let $\vec d^{out} = (d_1^{out},\dots,d_N^{out})$ and $\vec d^{in} = (d_1^{in},\dots,d_N^{in})$ be the out- and in-degree sequences of the network, where the coordinates $d_i^{out}$ and $d_i^{in}$ denote the out- and in-degree of node $i$, respectively.

Edges are assigned iteratively by a two-stage scheme. The first node $i$ is selected with probability proportional to its remaining degree:

$$
p(i) = \frac{d_i^{out}+d_i^{in}}{\sum_{k=1}^N (d_k^{out}+d_k^{in})}
$$

The second node $j$ is selected with probability conditioned on node $i$:

$$
p(j)=\frac{p_{ij}}{\sum_{k=1}^N p_{ik}}
$$

Once the edge $(i,j)\in\mathcal{E}$ is assigned, the degrees $d_i^{out}$ and $d_j^{in}$ are decremented, the probability $p_{ij}=0$ is updated, and the edge is masked from ocurring as a multiedge.

If it is found that any $d_i^{out}=0$, the probabilities are updated to $p_{ij}=0$ for all $j$. If it is found that any $d_i^{in}=0$, the probabilities are updated to $p_{ji}=0$ for all $j$. If the probability matrix $P=0$ but the sum of the remaining degrees $\sum_{k=1}^N (d_k^{out}+d_k^{in}) \neq 0$, the algorithm defaults to the directed configuration model to randomly assign self-loops and multiedges.

## **Functions**

### [**sdcdp_model**](#sdcdp_model)

`sdcdp_model(degrees, probabilities, directed=False, simple=False)`

Generates a network based on the Social Distance Configuration model with Degree Preservation (SDC-DP).

| Parameters        |                           | |
|-------------------|---------------------------|-|
| **degrees** | ***array-like of int*** | If ***directed*** is ***False***, a degree sequence with shape `(N,)`. If ***directed*** is ***True***, an out- and in-degree sequence with shape `(2, N)`. |
| **probabilities** | ***array-like of float*** | Edge formation probabilities with shape `(N, N)`. If ***directed*** is ***False***, must be symmetric. If ***directed*** is ***True***, may be asymmetric. |
| **directed** | ***bool, optional*** | If ***False***, assumes the network is undirected. If ***True***, assumes the network is directed. Default is ***False***. |
| **simple** | ***bool, optional*** | If ***False***, may return self-loops or multiedges. If ***True***, does not return self-loops or multiedges. Default is ***False***. |

| Returns | | |
|---------|-|-|
| **G** | ***nx.MultiGraph or nx.MultiDiGraph*** | Generated network matching input constraints. |

| Notes | |
|-------|-|
| **1.** | If ***directed*** is ***False***, the **degrees** parameter corresponds to a degree sequence, which must have an even sum. If ***directed*** is ***True***, the **degrees** parameter corresponds to an out-degree sequence in the first position, **degrees[0]**, and an in-degree sequence in the second position, **degrees[1]**, which must have equal sums. |
| **2.** | If ***directed*** is ***False***, the **probabilities** parameter must be symmetric, where entry $(i, j)$ corresponds to the probability of an edge between node $i$ and node $j$. If ***directed*** is ***True***, the **probabilities** parameter may be asymmetric, where entry $(i, j)$ corresponds to the probability of an edge beginning at node $i$ and terminating at node $j$. |
| **3.** | If ***simple*** is ***False***, the returned network may include self-loops or multiedges, but the target degree sequence will be satisfied. If ***simple*** is ***True***, the returned network will not include any self-loops or multiedges, but the target degree sequence may not be satisfied. |

## **Examples**

### **Undirected Network**

```python
import networkx as nx
import numpy as np
import sdcdp

# Step 1: Specify degree sequence and edge formation probabilities

degrees = np.array([2, 2, 2, 2, 2])

probabilities = np.array([
    [0.0, 0.8, 0.2, 0.5, 0.1],
    [0.8, 0.0, 0.3, 0.4, 0.2],
    [0.2, 0.3, 0.0, 0.6, 0.7],
    [0.5, 0.4, 0.6, 0.0, 0.9],
    [0.1, 0.2, 0.7, 0.9, 0.0]
])

# Step 2: Generate undirected network and allow self-loops and multiedges

G = sdcdp.sdcdp.sdcdp_model(degrees, probabilities, directed=False, simple=False)

# Step 3: Inspect edges

print("Edges:", list(G.edges()))
```

### **Directed Network**

```python
import networkx as nx
import numpy as np
import sdcdp

# Step 1: Specify degree sequences and edge formation probabilities

out_degrees = np.array([1, 2, 1, 1, 0])
in_degrees  = np.array([1, 1, 1, 1, 1])
degrees = np.stack([out_degrees, in_degrees])

probabilities = np.array([
    [0.0, 0.9, 0.1, 0.0, 0.0],
    [0.2, 0.0, 0.5, 0.3, 0.0],
    [0.4, 0.1, 0.0, 0.5, 0.0],
    [0.3, 0.2, 0.6, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])

# Step 2: Generate directed network and suppress self-loops and multiedges

G = sdcdp.sdcdp.sdcdp_model(degrees, probabilities, directed=True, simple=True)

# Step 3: Inspect edges

print("Edges:", list(G.edges()))
```

## **References**

Alec Fluer, Ian Laga, Logan Graham, Ellen Almirol, Makenna Meyer, and Breschine Cummins.  
From survey data to social multiplex models: Incorporating interlayer correlation from multiple data sources. In preparation.