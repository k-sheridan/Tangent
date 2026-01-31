"""
Core cppyy initialization and header loading.
"""
import cppyy
import os
from pathlib import Path

_initialized = False
_tangent_root = None


def get_tangent_root() -> Path:
    """Get the root directory of the Tangent project."""
    global _tangent_root
    if _tangent_root is None:
        # Navigate from tangent_py package to project root
        # python/tangent_py/core.py -> python/tangent_py -> python -> Tangent
        _tangent_root = Path(__file__).parent.parent.parent
    return _tangent_root


def init():
    """
    Initialize cppyy with Tangent headers.

    This must be called before using any Tangent functionality.
    It loads Eigen, Sophus, and all Tangent headers.

    Safe to call multiple times (idempotent).
    """
    global _initialized
    if _initialized:
        return

    root = get_tangent_root()

    # Enable -O3 optimization for LLVM auto-vectorization
    # This must be set before any code is compiled
    cppyy.gbl.gInterpreter.ProcessLine(".O3")

    # Add include paths
    cppyy.add_include_path(str(root / "include"))
    cppyy.add_include_path(str(root / "extern" / "eigen"))
    cppyy.add_include_path(str(root / "extern" / "sophus"))

    # Eigen configuration for cppyy compatibility
    # EIGEN_DONT_VECTORIZE disables Eigen's explicit SIMD intrinsics which cppyy's
    # Cling can't parse (especially ARM NEON intrinsics on aarch64).
    # With -O3, LLVM will still auto-vectorize the code.
    cppyy.cppdef("""
    #define EIGEN_DONT_VECTORIZE
    #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
    #define EIGEN_NO_DEBUG
    """)

    # Include Eigen
    cppyy.include("Eigen/Dense")
    cppyy.include("Eigen/Sparse")

    # Include Sophus
    cppyy.include("sophus/se3.hpp")
    cppyy.include("sophus/so3.hpp")

    # Include Tangent core headers
    cppyy.include("tangent/Differentiation/Jet.h")
    cppyy.include("tangent/Differentiation/JetTraits.h")
    cppyy.include("tangent/Variables/OptimizableVariable.h")
    cppyy.include("tangent/Variables/SimpleScalar.h")
    cppyy.include("tangent/Variables/SE3.h")
    cppyy.include("tangent/Variables/SO3.h")
    cppyy.include("tangent/Variables/InverseDepth.h")
    cppyy.include("tangent/ErrorTerms/ErrorTermBase.h")
    cppyy.include("tangent/ErrorTerms/AutoDiffErrorTerm.h")
    cppyy.include("tangent/Containers/SlotMap.h")
    cppyy.include("tangent/Types/GaussianPrior.h")
    cppyy.include("tangent/Types/SparseBlockMatrix.h")
    cppyy.include("tangent/Optimization/OptimizerContainers.h")
    cppyy.include("tangent/Optimization/HuberLossFunction.h")
    cppyy.include("tangent/Optimization/PSDSchurSolver.h")
    cppyy.include("tangent/Optimization/Marginalizer.h")
    cppyy.include("tangent/Optimization/SSEOptimizer.h")

    _initialized = True


def define_error_term(cpp_code: str):
    """
    JIT compile a C++ error term class.

    The code should define a class inheriting from AutoDiffErrorTerm.

    Args:
        cpp_code: C++ code string defining the error term class

    Example:
        >>> define_error_term('''
        ... class DiffError : public Tangent::AutoDiffErrorTerm<DiffError, double, 1,
        ...                                                      Tangent::SimpleScalar,
        ...                                                      Tangent::SimpleScalar> {
        ... public:
        ...     DiffError(Tangent::VariableKey<Tangent::SimpleScalar> k1,
        ...               Tangent::VariableKey<Tangent::SimpleScalar> k2) {
        ...         std::get<0>(variableKeys) = k1;
        ...         std::get<1>(variableKeys) = k2;
        ...         information.setIdentity();
        ...     }
        ...
        ...     template <typename T, typename V1, typename V2>
        ...     Eigen::Matrix<T, 1, 1> computeError(const V1& v1, const V2& v2) const {
        ...         Eigen::Matrix<T, 1, 1> err;
        ...         err(0) = v2 - v1;
        ...         return err;
        ...     }
        ... };
        ... ''')
    """
    init()
    cppyy.cppdef(cpp_code)


def is_initialized() -> bool:
    """Check if cppyy has been initialized."""
    return _initialized
