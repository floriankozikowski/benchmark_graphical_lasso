from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):

    name = "graphical_lasso"

    url = "https://github.com/benchopt/benchmark_graphical_lasso"

    requirements = ["numpy"]

    min_benchopt_version = "1.5"

    parameters = {
        'alpha': [0.01, 0.1, 0.5],
    }

    def __init__(self, alpha=0.1):
        # Store the parameters of the objective
        self.alpha = alpha

    def set_data(self, X, cov_true=None):
        """Set the data for the graphical lasso problem."""
        self.X = X
        self.cov_true = cov_true
        self.n_samples, self.n_features = X.shape

        # Compute empirical covariance matrix
        self.S = np.cov(X.T, bias=True)

    def get_objective(self):
        """Return the data needed by the solvers."""
        return dict(
            X=self.X,
            S=self.S,
            alpha=self.alpha,
            n_samples=self.n_samples,
            n_features=self.n_features
        )

    def evaluate_result(self, precision):
        """Evaluate the precision matrix estimate."""
        # Graphical lasso objective: -log det(Theta) + tr(S * Theta) + alpha ||Theta||_1
        log_det = np.linalg.slogdet(precision)[1]
        trace_term = np.trace(self.S @ precision)
        l1_penalty = self.alpha * np.sum(np.abs(precision))

        objective_value = -log_det + trace_term + l1_penalty

        results = {'value': objective_value}

        # Basic sparsity metric
        results['nnz'] = np.sum(np.abs(precision) > 1e-8)

        # Error metrics if ground truth is available
        if self.cov_true is not None:
            precision_true = np.linalg.inv(self.cov_true)

            # Frobenius error (absolute)
            results['frobenius_error'] = np.linalg.norm(
                precision - precision_true, 'fro'
            )

            # Normalized accuracy (similar to GGLasso's benchmarking approach)
            precision_true_norm = np.linalg.norm(precision_true, 'fro')
            if precision_true_norm > 0:
                results['normalized_error'] = (
                    results['frobenius_error'] / precision_true_norm
                )
            else:
                results['normalized_error'] = 0.0

        return results

    def get_one_result(self):
        """Return a dummy result for testing."""
        return dict(precision=np.eye(self.n_features))
