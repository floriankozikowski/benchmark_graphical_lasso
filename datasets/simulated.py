from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import networkx as nx
    import numpy as np


class Dataset(BaseDataset):
    """Synthetic scale‑free graphs for the Graphical‑Lasso benchmark.

    This follows the recipe used in the original gglasso benchmark:
    1. Build a Barabási–Albert/scale‑free graph with NetworkX.
    2. Assign random weights in the range [-0.5, 0.5] to each edge.
    3. Symmetrise and boost the diagonal by +0.1 to guarantee SPD.
    """

    name = "simulated"
    parameters = {
        "p": [500, 1000, 2000, 4000],
        "density": [0.02, 0.04],
        "seed": [27],
    }
    test_parameters = {"p": [50]}
    requirements = ["numpy", "networkx", "scipy"]

    def __init__(self, p=100, density=0.02, seed=27):
        self.p = p
        self.density = density
        self.seed = seed
        # Keep the same ratio n_samples ≈ p + 50 as in the gglasso notebook.
        self.n_samples = p + 50

    def get_data(self):
        rng = np.random.default_rng(self.seed)
        theta_true = self._generate_powerlaw_precision(
            self.p, self.density, rng
        )
        sigma_true = np.linalg.inv(theta_true)
        X = rng.multivariate_normal(
            np.zeros(self.p), sigma_true, size=self.n_samples
        )
        return dict(X=X, cov_true=sigma_true)

    def _generate_powerlaw_precision(self, p, density, rng):
        G = nx.scale_free_graph(p, seed=rng)
        A_bool = nx.to_numpy_array(G) > 0
        mask = rng.random(size=A_bool.shape) < density
        A_bool &= mask
        W = np.zeros_like(A_bool, dtype=float)
        W[A_bool] = rng.uniform(-0.5, 0.5, size=A_bool.sum())
        W_upper = np.triu(W, 1)
        theta = W_upper + W_upper.T
        np.fill_diagonal(theta, 1.0)
        theta += 0.1 * np.eye(p)
        return theta
