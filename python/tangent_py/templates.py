"""
Helper template for generating C++ error term code.
"""
from typing import List


def error_term_template(
    name: str,
    residual_dim: int,
    var_types: List[str],
    compute_body: str,
    extra_members: str = "",
    extra_constructor_params: str = "",
    extra_constructor_init: str = ""
) -> str:
    """
    Generate C++ code for an AutoDiffErrorTerm.

    Supports any combination of variable types, residual dimensions,
    and optional extra members/constructor parameters.

    Args:
        name: Class name for the error term
        residual_dim: Dimension of the residual vector
        var_types: List of variable type names (e.g., ["SimpleScalar", "SE3"])
        compute_body: C++ code for the body of computeError().
                     Use v0, v1, v2... to refer to variables.
                     Use err(i) to set residual components.
        extra_members: Additional class member declarations (e.g., "double target;")
        extra_constructor_params: Additional constructor parameters (e.g., "double t")
        extra_constructor_init: Additional constructor initialization code (e.g., "target = t;")

    Returns:
        C++ code string defining the error term class

    Examples:
        Binary constraint between two variables:

        >>> code = error_term_template(
        ...     name="DifferenceError",
        ...     residual_dim=1,
        ...     var_types=["SimpleScalar", "SimpleScalar"],
        ...     compute_body="err(0) = v1 - v0;"
        ... )

        Unary prior with a target value:

        >>> code = error_term_template(
        ...     name="ScalarPrior",
        ...     residual_dim=1,
        ...     var_types=["SimpleScalar"],
        ...     compute_body="err(0) = v0 - target;",
        ...     extra_members="double target;",
        ...     extra_constructor_params="double t",
        ...     extra_constructor_init="target = t;"
        ... )
    """
    # Build variable type list for template
    var_list = ", ".join(f"Tangent::{v}" for v in var_types)

    # Build constructor parameter list
    key_params = ", ".join(
        f"Tangent::VariableKey<Tangent::{v}> k{i}"
        for i, v in enumerate(var_types)
    )
    if extra_constructor_params:
        key_params += ", " + extra_constructor_params

    # Build key assignments
    key_assigns = "\n        ".join(
        f"std::get<{i}>(variableKeys) = k{i};"
        for i in range(len(var_types))
    )

    # Build template parameters for computeError
    template_params = ", ".join(f"typename V{i}" for i in range(len(var_types)))
    func_params = ", ".join(f"const V{i}& v{i}" for i in range(len(var_types)))

    # Format extra_members with proper indentation
    members_block = ""
    if extra_members:
        members_block = f"\n    {extra_members}\n"

    # Format extra_constructor_init
    init_block = ""
    if extra_constructor_init:
        init_block = f"\n        {extra_constructor_init}"

    return f"""
class {name} : public Tangent::AutoDiffErrorTerm<{name}, double, {residual_dim}, {var_list}> {{
public:{members_block}
    {name}({key_params}) {{
        {key_assigns}
        information.setIdentity();{init_block}
    }}

    template <typename T, {template_params}>
    Eigen::Matrix<T, {residual_dim}, 1> computeError({func_params}) const {{
        Eigen::Matrix<T, {residual_dim}, 1> err;
        {compute_body}
        return err;
    }}
}};
"""
