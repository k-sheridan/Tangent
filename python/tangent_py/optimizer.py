"""
Optimizer wrapper for Tangent SSEOptimizer.
"""
import cppyy
from typing import List, Optional
from dataclasses import dataclass
from .core import init


@dataclass
class OptimizationResult:
    """Result of an optimization run.

    Attributes:
        iterations: Number of optimizer iterations performed.
        initial_error: Total squared error before optimization.
        final_error: Total squared error after optimization.
        converged: Whether the optimizer converged to a minimum.
        error_decreased: Whether the final error is less than the initial error.
        error_history: List of total error values at each iteration.
    """
    iterations: int
    initial_error: float
    final_error: float
    converged: bool
    error_decreased: bool
    error_history: List[float]


class Optimizer:
    """
    Nonlinear least squares optimizer using Tangent's SSEOptimizer.

    This class wraps SSEOptimizer and handles the complex template
    instantiation required by cppyy.

    Example:
        >>> opt = Optimizer(
        ...     variables=["SimpleScalar"],
        ...     error_terms=["DifferenceError"]
        ... )
        >>> x = SimpleScalar(10.0)
        >>> y = SimpleScalar(0.0)
        >>> k1 = opt.add_variable(x)
        >>> k2 = opt.add_variable(y)
        >>> opt.add_error_term("DifferenceError", k1, k2)
        >>> result = opt.optimize()
    """

    # Class-level counter for unique type names
    _instance_counter = 0

    def __init__(self,
                 variables: List[str],
                 error_terms: List[str],
                 huber_delta: float = 1000.0):
        """
        Create an optimizer.

        Args:
            variables: List of variable type names (e.g., ["SimpleScalar", "SE3"])
            error_terms: List of error term class names
            huber_delta: Huber loss function delta parameter
        """
        init()

        self._var_types = variables
        self._err_types = error_terms

        # Get unique instance ID
        Optimizer._instance_counter += 1
        self._instance_id = Optimizer._instance_counter

        # Build the fully-qualified type strings
        var_group = ", ".join(f"Tangent::{v}" for v in variables)
        err_group = ", ".join(error_terms)

        # Store for later use
        self._var_group_str = var_group
        self._err_group_str = err_group

        # Build the complex nested optimizer type
        optimizer_type_str = self._build_optimizer_type_string(var_group, err_group)
        solver_type_str = self._build_solver_type_string(var_group, err_group)

        # Use cppyy to instantiate the optimizer type and create an instance
        try:
            # Define a factory function in C++ to create the optimizer
            # The solver needs a loss function reference, so we use a static loss function
            factory_code = f"""
            namespace TangentPy {{
                using OptType_{self._instance_id} = {optimizer_type_str};
                using SolverType_{self._instance_id} = {solver_type_str};

                // Static loss function that persists for the lifetime of the optimizer
                static Tangent::HuberLossFunction<double> lossFunction_{self._instance_id}({huber_delta});

                OptType_{self._instance_id}* create_optimizer_{self._instance_id}() {{
                    SolverType_{self._instance_id} solver(lossFunction_{self._instance_id});
                    return new OptType_{self._instance_id}(std::move(solver));
                }}
            }}
            """
            cppyy.cppdef(factory_code)

            # Create the optimizer instance
            factory = getattr(cppyy.gbl.TangentPy, f"create_optimizer_{self._instance_id}")
            self._cpp_opt = factory()

        except Exception as e:
            raise RuntimeError(f"Failed to instantiate optimizer: {e}\n"
                             f"Type string: {optimizer_type_str}")

    def _build_solver_type_string(self, var_group: str, err_group: str) -> str:
        """Build the PSDSchurSolver template type string."""
        return f"""Tangent::PSDSchurSolver<
            Tangent::Scalar<double>,
            Tangent::LossFunction<Tangent::HuberLossFunction<double>>,
            Tangent::ErrorTermGroup<{err_group}>,
            Tangent::VariableGroup<{var_group}>,
            Tangent::VariableGroup<>
        >"""

    def _build_optimizer_type_string(self, var_group: str, err_group: str) -> str:
        """Build the full SSEOptimizer template type string."""
        solver_type = self._build_solver_type_string(var_group, err_group)
        return f"""Tangent::SSEOptimizer<
            Tangent::Scalar<double>,
            {solver_type},
            Tangent::GaussianPrior<
                Tangent::Scalar<double>,
                Tangent::VariableGroup<{var_group}>
            >,
            Tangent::Marginalizer<
                Tangent::Scalar<double>,
                Tangent::VariableGroup<{var_group}>,
                Tangent::ErrorTermGroup<{err_group}>
            >,
            Tangent::VariableGroup<{var_group}>,
            Tangent::ErrorTermGroup<{err_group}>
        >"""

    def add_variable(self, variable, prior_info=None):
        """
        Add a variable to the optimization problem.

        Args:
            variable: A Tangent variable instance (SimpleScalar, SE3, etc.)
            prior_info: Optional prior information matrix (numpy array or scalar)

        Returns:
            Variable key for referencing this variable
        """
        key = self._cpp_opt.addVariable(variable)

        if prior_info is not None:
            self.set_prior(key, prior_info)

        return key

    def set_prior(self, key, information):
        """
        Set the prior information matrix for a variable.

        Args:
            key: Variable key from add_variable()
            information: Information matrix (scalar for 1D, or numpy array)
        """
        import numpy as np

        # Convert scalar to Eigen matrix if needed
        if np.isscalar(information):
            # Get dimension from variable type
            var = self._cpp_opt.getVariable(key)
            dim = var.dimension

            # Create Eigen matrix
            info_matrix = cppyy.gbl.Eigen.Matrix["double", dim, dim]()
            info_matrix.setIdentity()

            # Scale by the scalar value
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        info_matrix[i, j] = float(information)
                    else:
                        info_matrix[i, j] = 0.0

            self._cpp_opt.setVariablePrior(key, info_matrix)
        else:
            # Assume it's already an Eigen matrix or numpy array
            self._cpp_opt.setVariablePrior(key, information)

    def add_error_term(self, error_type: str, *args):
        """
        Add an error term to the optimization problem.

        Args:
            error_type: Name of the error term class
            *args: Arguments to pass to the error term constructor
                   (typically variable keys and measurement data)

        Returns:
            Error term key
        """
        # Get the error term class from cppyy
        err_class = getattr(cppyy.gbl, error_type)

        # Construct the error term with provided arguments
        err = err_class(*args)

        # Add to optimizer
        return self._cpp_opt.addErrorTerm(err)

    def get_variable(self, key):
        """
        Get a variable by its key.

        Args:
            key: Variable key from add_variable()

        Returns:
            The variable instance with current optimized value
        """
        return self._cpp_opt.getVariable(key)

    @property
    def settings(self):
        """Access optimizer settings."""
        return self._cpp_opt.settings

    def optimize(self, max_iterations: Optional[int] = None) -> OptimizationResult:
        """
        Run the optimization.

        Args:
            max_iterations: Override the default maximum iterations

        Returns:
            OptimizationResult with convergence information
        """
        if max_iterations is not None:
            self._cpp_opt.settings.maximumIterations = max_iterations

        cpp_result = self._cpp_opt.optimize()

        return OptimizationResult(
            iterations=cpp_result.iterations,
            initial_error=cpp_result.initialError,
            final_error=cpp_result.finalError,
            converged=cpp_result.converged,
            error_decreased=cpp_result.errorDecreased,
            error_history=list(cpp_result.errorHistory)
        )

    def variable_count(self, var_type: str) -> int:
        """Get the number of variables of a given type."""
        var_class = getattr(cppyy.gbl.Tangent, var_type)
        return self._cpp_opt.variableCount[var_class]()

    def error_term_count(self, error_type: str) -> int:
        """Get the number of error terms of a given type."""
        err_class = getattr(cppyy.gbl, error_type)
        return self._cpp_opt.errorTermCount[err_class]()
