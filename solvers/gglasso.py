from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from gglasso.problem import glasso_problem


class Solver(BaseSolver):
    name = 'gglasso'
    requirements = ['gglasso']

    def set_objective(self, X, S, alpha, n_samples, n_features):
        """Set the objective data for the solver."""
        self.S = S
        self.n_samples = n_samples
        self.alpha = alpha

    def run(self, n_iter):
        """Run the GGLasso using the glasso_problem interface."""
        # Create single graphical lasso problem
        problem = glasso_problem(
            S=self.S,
            N=self.n_samples,
            reg_params={'lambda1': self.alpha}
        )

        # Solve the problem
        problem.solve()

        # Store the precision matrix
        self.precision_matrix = problem.solution.precision_

    def get_result(self):
        """Return the precision matrix."""
        return dict(precision=self.precision_matrix)

# TODO: add block-wise solver
