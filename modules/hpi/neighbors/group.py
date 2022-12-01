from typing import Tuple, Dict, Optional, Callable
import numpy as np
from scipy.sparse import csgraph 
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph


METHOD_NEIGHBORS_RADIUS = "radius"
METHOD_NEIGHBORS_K = "kneighbors"


def _group_by_radius(
    X: np.ndarray,
    radius: float = 0.05,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: int = 2,
    metric_params: Optional[Dict] = None,
    include_self: bool = False,
    n_jobs: Optional[int] = None,
) -> Tuple[int, Dict[int, int]]:
    """Group features.
    
    Parameters
    ----------
        X : (N, M), ndarray,
            Feature vectors to group up,
            where N is number of classes and M is number of features
        radius : float,
            Threshold radius to group up feature vectors.
        See references for more details.

    Returns 
    -------
        (n_components, gourp): Tuple[int, Dict[int, int]]
            n_components : int 
                Number of connected components detected. 
            group : dict 
                key-value pair where key represents unique classes,
                and value reparesenting minimum class index in the same group (CC). 
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    """
    if X.shape[0] > 1000:
        raise ValueError(f"Too many classes to handle ({X.shape[0]}). It must be less than 1000.")
    
    adj = radius_neighbors_graph(X, radius, mode, metric, p, metric_params, include_self, n_jobs)
    n_components, labels = csgraph.connected_components(adj, directed=False)
    group = dict(zip(np.arange(X.shape[0]), labels))
    return (n_components, group)


def _group_by_kneighbors(
    X: np.ndarray,
    n_neighbors: int,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: int = 2,
    metric_params: Optional[Dict] = None,
    include_self: bool = False,
    n_jobs: Optional[int] = None,
) -> Tuple[int, Dict[int, int]]:
    """Group features.
    
    Parameters
    ----------
        X : (N, M), ndarray,
            Feature vectors to group up,
            where N is number of classes and M is number of features
        n_neighbors : int,
            Threshold neighbors to group up feature vectors.
        See references for more details.

    Returns 
    -------
        (n_components, gourp): Tuple[int, Dict[int, int]]
            n_components : int 
                Number of connected components detected. 
            group : dict 
                key-value pair where key represents unique classes,
                and value reparesenting minimum class index in the same group (CC). 
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    """
    if X.shape[0] > 1000:
        raise ValueError(f"Too many classes to handle ({X.shape[0]}). It must be less than 1000.")
    
    adj = kneighbors_graph(X, n_neighbors , mode, metric, p, metric_params, include_self, n_jobs)
    n_components, labels = csgraph.connected_components(adj, directed=False)
    group = dict(zip(np.arange(X.shape[0]), labels))
    return (n_components, group)


METHODS: Dict[str, Callable] = {
    METHOD_NEIGHBORS_RADIUS: _group_by_radius,
    METHOD_NEIGHBORS_K: _group_by_kneighbors
}
def group(
    X: np.ndarray,
    method=METHOD_NEIGHBORS_RADIUS,
    **kwargs
):
    """Group features."""
    return METHODS[method](X, **kwargs)