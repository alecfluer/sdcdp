import numbers
import warnings

import numpy as np
import pandas as pd
import scipy

class MultiplexSDA:
    """
    The Social Distance Attachment (SDA) method generalized to multiplex networks.
    
    See docs/sda.md for documentation and full mathematical details.
    
    Attributes
    ----------
    networks_dataframe : pd.DataFrame
        Stores characteristic distance, homophily, and weight for (network layer, social feature) pairs.
    features_dataframe : pd.DataFrame
        Stores node locations and distance function for social features.

    Properties
    ----------
    dataframe : pd.DataFrame
        Returns a merged view of `networks_dataframe` and `features_dataframe`.
    networks : np.ndarray
        Returns an array of unique network layer names.
    features : np.ndarray
        Returns an array of unique social feature names.
    number_of_networks : int or None
        Returns the number of unique network layers.
    number_of_features : int or None
        Returns the number of unique social features.
    number_of_nodes : int or None
        Returns the number of nodes inferred from the first assigned social feature.
    shape : tuple
        Returns `(number_of_networks, number_of_features, number_of_nodes)`.
    """
    
    def __init__(self):
        """Initialize internal dataframes for network layers and social features."""
        self.networks_dataframe = pd.DataFrame(columns=["network", "feature", "char_dist", "homophily", "weight"])
        self.features_dataframe = pd.DataFrame(columns=["feature", "locations", "dist_func"])

    def __repr__(self):
        """Return summary string with counts of network layers, social features, and nodes."""
        return (f"MultiplexSDA("
                f"Network Layers: {self.number_of_networks}, "
                f"Social Features: {self.number_of_features}, "
                f"Nodes: {self.number_of_nodes})")
    
    @property
    def dataframe(self):
        """Returns a merged view of `networks_dataframe` and `features_dataframe`."""
        dataframe = self.networks_dataframe.merge(self.features_dataframe, on="feature", how="left")
        return dataframe.sort_values("network").reset_index(drop=True)

    @property
    def networks(self):
        """Returns an array of unique network layer names."""
        return self.networks_dataframe["network"].dropna().unique()
    
    @property
    def features(self):
        """Returns an array of unique social feature names."""
        return self.features_dataframe["feature"].dropna().unique()

    @property
    def number_of_networks(self):
        """Returns the number of unique network layers."""
        if len(self.networks) > 0:
            return len(self.networks)
        return None
    
    @property
    def number_of_features(self):
        """Returns the number of unique social features."""
        if len(self.features) > 0:
            return len(self.features)
        return None
    
    @property
    def number_of_nodes(self):
        """Returns the number of nodes inferred from the first assigned social feature."""
        for _, row in self.features_dataframe.iterrows():
            if isinstance(row["locations"], (list, np.ndarray)):
                return len(row["locations"])
        return None
    
    @property
    def shape(self):
        """Returns `(number_of_networks, number_of_features, number_of_nodes)`."""
        return (self.number_of_networks, self.number_of_features, self.number_of_nodes)
    
    def _reset_indices(self):
        """Sort and reindex internal dataframes."""
        self.networks_dataframe.sort_values(["network", "feature"], inplace=True)
        self.networks_dataframe.reset_index(drop=True, inplace=True)
        self.features_dataframe.sort_values("feature", inplace=True)
        self.features_dataframe.reset_index(drop=True, inplace=True)
    
    def add_networks_from(self, networks):
        """
        Adds one or more network layers to the model.

        Each new network layer is paired with all existing social features.
        If no social features exist, placeholder rows are added with `np.nan` values.

        Parameters
        ----------
        networks : array-like of str
            Names of the network layers to add.
        """

        if not isinstance(networks, (list, tuple, np.ndarray)):
            raise TypeError("Invalid Input: 'networks' must be a list, tuple, or NumPy array.")
        for network in networks:
                if not isinstance(network, str):
                    raise TypeError(f"Invalid Input: network '{network}' must be a string.")

        network_rows = []
        for network in networks:
            if network not in self.networks:
                if len(self.features) > 0:
                    network_rows.extend((network, feature, np.nan, np.nan, np.nan) for feature in self.features)
                else:
                    network_rows.append((network, np.nan, np.nan, np.nan, np.nan))

        df = pd.DataFrame(network_rows, columns=self.networks_dataframe.columns)
        self.networks_dataframe = pd.concat([self.networks_dataframe, df], ignore_index=True)
        self.networks_dataframe.dropna(subset=["network"], inplace=True)

        self._reset_indices()
    
    def add_features_from(self, features):
        """
        Adds one or more social features to the model.

        Each new social feature is paired with all existing network layers.
        If no network layers exist, placeholder rows are added with `np.nan` values.

        Parameters
        ----------
        features : array-like of str
            Names of the social features to add.
        """

        if not isinstance(features, (list, tuple, np.ndarray)):
            raise TypeError("Invalid Input: 'features' must be a list, tuple, or NumPy array.")
        for feature in features:
            if not isinstance(feature, str):
                raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")

        feature_rows = []
        network_rows = []
        for feature in features:
            if feature not in self.features:
                feature_rows.append((feature, np.nan, np.nan))
                if len(self.networks) > 0:
                    network_rows.extend((network, feature, np.nan, np.nan, np.nan) for network in self.networks)
                else:
                    network_rows.append((np.nan, feature, np.nan, np.nan, np.nan))

        df = pd.DataFrame(feature_rows, columns=self.features_dataframe.columns)
        self.features_dataframe = pd.concat([self.features_dataframe, df], ignore_index=True)

        df = pd.DataFrame(network_rows, columns=self.networks_dataframe.columns)
        self.networks_dataframe = pd.concat([self.networks_dataframe, df], ignore_index=True)
        self.networks_dataframe.dropna(subset=["feature"], inplace=True)

        self._reset_indices()
    
    def remove_networks_from(self, networks):
        """
        Removes one or more network layers from the model.

        Each removed network layer is deleted from all existing social features.

        Parameters
        ----------
        networks : array-like of str
            Names of the network layers to remove.
        """

        if not isinstance(networks, (list, tuple, np.ndarray)):
            raise TypeError("Invalid Input: 'networks' must be a list, tuple, or NumPy array.")
        for network in networks:
                if not isinstance(network, str):
                    raise TypeError(f"Invalid Input: network '{network}' must be a string.")

        self.networks_dataframe = self.networks_dataframe[~self.networks_dataframe["network"].isin(networks)]

        if len(self.features) > 0 and self.networks_dataframe.empty:
            network_rows = [(np.nan, feature, np.nan, np.nan, np.nan) for feature in self.features]
            self.networks_dataframe = pd.DataFrame(network_rows, columns=self.networks_dataframe.columns)
        
        self._reset_indices()
    
    def remove_features_from(self, features):
        """
        Removes one or more social features from the model.

        Each removed social feature is deleted from all existing network layers.

        Parameters
        ----------
        features : array-like of str
            Names of the social features to remove.
        """

        if not isinstance(features, (list, tuple, np.ndarray)):
            raise TypeError("Invalid Input: 'features' must be a list, tuple, or NumPy array.")
        for feature in features:
            if not isinstance(feature, str):
                raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")
        
        networks = self.networks.copy()
        self.features_dataframe = self.features_dataframe[~self.features_dataframe["feature"].isin(features)]
        self.networks_dataframe = self.networks_dataframe[~self.networks_dataframe["feature"].isin(features)]
        
        if len(networks) > 0 and self.networks_dataframe.empty:
            network_rows = [(network, np.nan, np.nan, np.nan, np.nan) for network in networks]
            self.networks_dataframe = pd.DataFrame(network_rows, columns=self.networks_dataframe.columns)

        self._reset_indices()
    
    def assign_network_params(self, network, feature, char_dist, homophily, weight, create=False):
        """
        Assigns characteristic distance, homophily, and weight to a specified (network layer, social feature) pair.

        Parameters
        ----------
        network : str
            Name of the network layer for parameter assignment.
        
        feature : str
            Name of the social feature for parameter assignment.
        
        char_dist : float
            The characteristic distance. Must be greater than 0.
        
        homophily : float
            The homophily parameter. Must be greater than or equal to 1.
        
        weight : float
            The weight coefficient. Must be greater than or equal to 0.
        
        create : bool, optional
            If True, adds network layer or social feature to the model if missing. Default is False.
        """

        if not isinstance(network, str):
            raise TypeError(f"Invalid Input: network '{network}' must be a string.")
        if not isinstance(feature, str):
            raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")

        if not isinstance(char_dist, numbers.Number) or char_dist <= 0:
            raise ValueError("Invalid Input: 'char_dist' must be numeric and greater than 0.")
        if not isinstance(homophily, numbers.Number) or homophily < 1:
            raise ValueError("Invalid Input: 'homophily' must be numeric and greater than or equal to 1.")
        if not isinstance(weight, numbers.Number) or weight < 0:
            raise ValueError("Invalid Input: 'weight' must be numeric and greater than or equal to 0.")

        if network not in self.networks or feature not in self.features:
            if not create:
                raise ValueError(
                    f"Invalid Assignment: network '{network}' or feature '{feature}' does not exist. "
                    f"Use 'add_networks_from()' or 'add_features_from()' to add manually, "
                    f"or set 'create=True' to add automatically."
                )
            self.add_networks_from([network]) if network not in self.networks else None
            self.add_features_from([feature]) if feature not in self.features else None
        
        idx = self.networks_dataframe.index[
            (self.networks_dataframe["network"] == network) &
            (self.networks_dataframe["feature"] == feature)
        ]

        self.networks_dataframe.at[idx[0], "char_dist"] = char_dist
        self.networks_dataframe.at[idx[0], "homophily"] = homophily
        self.networks_dataframe.at[idx[0], "weight"] = weight
        self._reset_indices()
    
    def assign_feature_params(self, feature, locations, dist_func, create=False):
        """
        Assigns node locations and distance function to a specified social feature.

        Parameters
        ----------
        feature : str
            Name of the social feature for parameter assignment.
        
        locations : array-like
            The node locations. Can be multidimensional with shape (N,) or (N, d).
        
        dist_func : callable
            The distance function. Must be compatible with the node locations.
        
        create : bool, optional
            If True, adds social feature to the model if missing. Default is False.
        """

        if not isinstance(feature, str):
            raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")

        if not isinstance(locations, (list, np.ndarray)) or np.array(locations).ndim > 2:
            raise TypeError("Invalid Input: 'locations' must be a list or NumPy array with shape (N,) or (N, d).")
        if not callable(dist_func):
            raise TypeError("Invalid Input: 'dist_func' must be callable and compatible with 'locations'.")

        nodes = self.number_of_nodes
        if nodes is not None and nodes != len(locations):
            raise ValueError(
                f"Invalid Input: inconsistent node count. Expected {nodes} nodes based on previously assigned features. "
                f"Received {len(locations)} nodes. Ensure 'locations' is a list or NumPy array with shape (N,) or (N, d). "
                f"Use 'clear_feature_params_from()' to clear previously assigned features if necessary."
            )

        if feature not in self.features:
            if not create:
                raise ValueError(
                    f"Invalid Assignment: feature '{feature}' does not exist. "
                    f"Use 'add_features_from()' to add manually, "
                    f"or set 'create=True' to add automatically."
                )
            self.add_features_from([feature])
        
        idx = self.features_dataframe.index[
            self.features_dataframe["feature"] == feature
        ]

        self.features_dataframe.at[idx[0], "locations"] = locations
        self.features_dataframe.at[idx[0], "dist_func"] = dist_func
        self._reset_indices()
    
    def clear_network_params_from(self, networks=None, features=None):
        """
        Clears characteristic distance, homophily, and weight from specified (network layer, social feature) pairs.

        The method applies clearing across the Cartesian product of `networks` and `features`. If either is set to
        None, all network layers or social features are selected accordingly.

        Parameters
        ----------
        networks : array-like of str or None, optional
            Names of the network layers to clear.
            If None, all network layers are selected. Default is None.

        features : array-like of str or None, optional
            Names of the social features to clear.
            If None, all social features are selected. Default is None.
        """

        if networks is not None:
            if not isinstance(networks, (list, tuple, np.ndarray)):
                raise TypeError("Invalid Input: 'networks' must be a list, tuple, or NumPy array.")
            for network in networks:
                if not isinstance(network, str):
                    raise TypeError(f"Invalid Input: network '{network}' must be a string.")
        if features is not None:
            if not isinstance(features, (list, tuple, np.ndarray)):
                raise TypeError("Invalid Input: 'features' must be a list, tuple, or NumPy array.")
            for feature in features:
                if not isinstance(feature, str):
                    raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")

        mask = pd.Series(True, index=self.networks_dataframe.index)

        if networks is not None:
            mask &= self.networks_dataframe["network"].isin(networks)
        if features is not None:
            mask &= self.networks_dataframe["feature"].isin(features)

        self.networks_dataframe.loc[mask, ["char_dist", "homophily", "weight"]] = np.nan
        self._reset_indices()
    
    def clear_feature_params_from(self, features=None):
        """
        Clears node locations and distance function from specified social features.

        Parameters
        ----------
        features : array-like of str or None, optional
            Names of the social features to clear.
            If None, all social features are selected. Default is None.
        """

        if features is not None:
            if not isinstance(features, (list, tuple, np.ndarray)):
                raise TypeError("Invalid Input: 'features' must be a list, tuple, or NumPy array.")
            for feature in features:
                if not isinstance(feature, str):
                    raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")

        mask = pd.Series(True, index=self.features_dataframe.index)

        if features is not None:
            mask &= self.features_dataframe["feature"].isin(features)

        self.features_dataframe.loc[mask, ["locations", "dist_func"]] = np.nan
        self._reset_indices()
    
    def compute_dist_matrices(self, features=None):
        """
        Computes pairwise distances between nodes using node locations and distance function for each specified social feature.

        Parameters
        ----------
        features : array-like of str or None, optional
            Names of the social features to compute distances for.
            If None, all social features are selected. Default is None.

        Returns
        -------
        dist_matrices : dict[str, np.ndarray]
            Dictionary mapping each social feature name to a symmetric pairwise distance matrix
            with shape (N, N).
        """

        dist_matrices = {}
        nodes = self.number_of_nodes
        df = self.features_dataframe.dropna()

        if features is not None:
            if not isinstance(features, (list, tuple, np.ndarray)):
                raise TypeError("Invalid Input: 'features' must be a list, tuple, or NumPy array.")
            for feature in features:
                if not isinstance(feature, str):
                    raise TypeError(f"Invalid Input: feature '{feature}' must be a string.")
            df = df[df["feature"].isin(features)]

        for _, row in df.iterrows():
            feature = row["feature"]
            locations = row["locations"]
            dist_func = row["dist_func"]

            locations = np.array(locations)
            if locations.ndim == 1:
                locations = locations.reshape(-1, 1)

            dist_matrix = scipy.spatial.distance.pdist(locations, metric=dist_func)
            dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
            
            dist_matrices[feature] = dist_matrix
            if not np.array_equal(dist_matrix, dist_matrix.T):
                warnings.warn(f"Computed distance matrix for feature '{feature}' is not symmetric.", RuntimeWarning)
        
        requested_features = set(features) if features is not None else set(self.features)
        for feature in requested_features - set(df["feature"].unique()):
            dist_matrices[feature] = np.full((nodes, nodes), np.nan)
        
        return dist_matrices
    
    def compute_prob_matrices(self, networks=None):
        """
        Computes SDA connection probabilities between nodes using available class data for each specified network layer.

        Parameters
        ----------
        networks : array-like of str or None, optional
            Names of the network layers to compute probabilities for.
            If None, all network layers are selected. Default is None.

        Returns
        -------
        prob_matrices : dict[str, np.ndarray]
            Dictionary mapping each network layer name to a symmetric probability matrix
            with shape (N, N).
        """

        prob_matrices = {}
        nodes = self.number_of_nodes
        df = self.dataframe.dropna()

        if networks is not None:
            if not isinstance(networks, (list, tuple, np.ndarray)):
                raise TypeError("Invalid Input: 'networks' must be a list, tuple, or NumPy array.")
            for network in networks:
                if not isinstance(network, str):
                    raise TypeError(f"Invalid Input: network '{network}' must be a string.")
            df = df[df["network"].isin(networks)]
        
        dist_matrices = self.compute_dist_matrices(features=df["feature"].unique())

        for network in df["network"].unique():

            total_weight = 0
            prob_matrices[network] = np.zeros((nodes, nodes))
            network_feature_data = df[df["network"] == network]

            for _, row in network_feature_data.iterrows():
                feature = row["feature"]
                b = row["char_dist"]
                h = row["homophily"]
                w = row["weight"]

                if w == 0:
                    continue

                dist_matrix = dist_matrices[feature]
                prob_matrix = 1 / (1 + (dist_matrix / b) ** h)

                total_weight += w
                prob_matrices[network] += w * prob_matrix

            if total_weight > 0:
                prob_matrices[network] /= total_weight

            prob_matrix = prob_matrices[network]
            if not np.array_equal(prob_matrix, prob_matrix.T):
                warnings.warn(f"Computed probability matrix for network '{network}' is not symmetric.", RuntimeWarning)
        
        requested_networks = set(networks) if networks is not None else set(self.networks)
        for network in requested_networks - set(df["network"].unique()):
            prob_matrices[network] = np.full((nodes, nodes), np.nan)

        return prob_matrices