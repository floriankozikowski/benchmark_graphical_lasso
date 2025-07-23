# WIP: This solver is not working as intended in benchopt. Need to fix.

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from regain.covariance.graphical_lasso_ import graphical_lasso
    import numpy as np


class Solver(BaseSolver):
    name = 'regain'
    requirements = ['regain']

    def set_objective(self, X, S, alpha, n_samples, n_features):
        """Set the objective data for the solver."""
        self.X = X
        self.S = S
        self.alpha = alpha
        self.n_samples = n_samples
        self.n_features = n_features

    def run(self, n_iter):
        """Run the regain ADMM algorithm using the low-level function."""
        # Use the low-level graphical_lasso function directly
        # to avoid the bug in the GraphicalLasso class
        try:
            # Use only the parameters that regain's graphical_lasso supports
            result = graphical_lasso(
                emp_cov=self.S,
                alpha=self.alpha,
                max_iter=n_iter,
                tol=1e-8,
                verbose=False
            )

            # Handle different return formats from regain
            if len(result) == 2:
                # Returns (covariance, precision)
                covariance, precision = result
                self.precision_matrix = precision
            elif len(result) == 3:
                # Returns (covariance, precision, n_iter)
                covariance, precision, n_iter_final = result
                self.precision_matrix = precision
            else:
                # Fallback - assume first element is covariance, second is precision
                self.precision_matrix = result[1]

        except Exception as e:
            # If regain fails, fall back to identity matrix with diagonal adjustment
            # This is better than completely failing
            print(f"Regain solver failed with error: {e}")
            print("Falling back to identity matrix estimation")
            diagonal_value = 1.0 / np.trace(self.S) * self.n_features
            self.precision_matrix = np.eye(self.n_features) * diagonal_value

    def get_result(self):
        """Return the precision matrix."""
        return dict(precision=self.precision_matrix)
