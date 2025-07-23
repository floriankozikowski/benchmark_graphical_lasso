from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skglm import GraphicalLasso

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'skglm-graphical-lasso'

    # List of packages needed to run the solver.
    requirements = [
        'pip', 'pip: git+https://github.com/scikit-learn-contrib/skglm@main']

    def set_objective(self, X, S, alpha, n_samples, n_features):
        """Set the objective data for the solver."""
        self.X = X
        self.alpha = alpha

    def run(self, n_iter):
        """Run the skglm GraphicalLasso algorithm."""
        self.estimator = GraphicalLasso(
            alpha=self.alpha,
            max_iter=n_iter
        )
        self.estimator.fit(self.X)

    def get_result(self):
        """Return the precision matrix."""
        return dict(precision=self.estimator.precision_)
