# SSEOptimizer
## Overview

SSEOptimizer is a semi generic nonlinear optimizer capable of performing Levenberg-Marquardt or Gauss-Newton on manifold. SSEOptimizer also stores a gaussian prior for all variables in its state. The gaussian prior is stored in a sparse block format which allows it to handle a very large state dimensionality assuming it remains sparse.

SSEOptimizer was designed for is small SLAM problems. Specifically, SSEOptimizer was designed for windowed SLAM problems.

```cpp
/// Requirement: E1 != E2 != E3... and V1 != V2 != V3...
class SSEOptimizer <ScalarType, VariableGroup<V1, V2, ...>, ErrorTermGroup<E1, E2, ...>> {

public:

// Adds a variable to the problem. The returned variable key should be stored by the user for future use.
VariableKey<V> addVariable(V optimizableVariable);

// Updates the marginal information of the given variable.
void setVariablePrior(VariableKey<V>, Eigen::Matrix<ScalarType, V::dimension, V::dimension>& informationMatrix);

// Removes a variable and it's associated error terms, while approximating the 
// information they provided to the remaining variables with a gaussian prior.
void marginalizeVariable(VariableKey<V> key);

// Removes a variable from the problem without marginalizing it.
void removeVariable(VariableKey<V> key);

// Adds error term to the problem.
ErrorTermKey<E> addErrorTerm(E errorTerm);

// Removes error term from the problem.
void removeErrorTerm(ErrorTermKey<E> errorTermKey);

// Iteratively refines the variables until the SSE of the error terms is minimized.
void optimize();

private:

std::shared_ptr<ErrorTermContainer<E1, ...>> errorTerms; // stores all error term types.
std::shared_ptr<VariableContainer<V1, ...>> variables; // stores all variables.

PSDLinearSystem<VariableGroup<V1, V2, ...>, ErrorTermGroup<E1, E2, ...>> linearSystem; // Used to solve for perturbation.
GaussianPrior<VariableGroup<V1, V2, ...>> prior; // Used during marginalization to approximate deleted information.

};
```
## Variable and ErrorTerm Storage
To keep the SSEOptimizer general, I allow the user to specify the types of variables and error terms in a template.
These types are then stored in tuples of slot maps which allows for easy access internally, and allows the user to interact
with the SSE optimizer to define their own marginalization strategy.
```cpp
struct VariableContainer : tuple<slot_map<V1>, ...> {
   // Gets the slot map for a variable
   slot_map<V> getVariableMap<V>();

   // Gets the index of the first scalar of the given variable. 
   // This is used to build and operate on a matrix.
   size_t variableIndex(VariableKey<V> key);
   
   // Computes the full dimensionality of all variables.
   size_t totalDimensions();
};
```
```cpp
struct ErrorTermContainer : tuple<slot_map<E1>, ...> {

};
```

## Solving for the perturbation.
During each iteration, the SSEOptimizer must set up a large linear system using the prior and linearlized error terms
to solve for a small perturbation which can be applied to the variables to minimize the SSE of the error terms give a prior estimate.
### Positive Semi-Definite Linear System
```cpp
class PSDLinearSystem<ScalarType, VariableGroup<V1, V2, ...>> {

  Matrix A;
  Vector dx, b;
  std::shared_ptr<VariableContainer>; // Gives the LinearSystem shared ownership with the variables.
  std::shared_ptr<ErrorTermContainer>; // Gives the LinearSystem shared access to the error term container.
  
  PSDLinearSystem(std::shared_ptr<VariableContainer> variables, std::shared_ptr<ErrorTermContainer> errorTerms);

  // zeros A and b
  void setZero();
  
  // resizes the linear system. A \in newSize x newSize. b \in newSize x 1. where newSize is the total variable dimension.
  void resize();
  
  // Sets up the linear system efficiently. (A_prior + Jt * W^(-1) * J) * dx = (b_prior + Jt * W^(-1) * e).
  void buildProblem(GaussianPrior<ScalarType, VariableGroup<V1, ...>>& prior);

  // Solves the linear system efficiently using schur complement.
  // The given index is used to find where the top left element of D is.
  // [ A   |  B ]
  // [ -------- ]
  // [ Bt  |  D ]
  // Where D is the block diagonal matrix.
  void solve();

  // overloaded += operator which adds two PSDLinearSystems together: (A1 + A2) * x = (b1 + b2)

};
```

## Marginalization
Before an error term or variable can be removed, the information it provided to the current solution is 
approximated with a quadratic error term / gaussian prior. Typically, this quadratic error term's information matrix is 
very sparse. To improve speed, the information matrix, A0, is stored as a sparse block matrix.
### Gaussian Prior
```cpp
class GaussianPrior<ScalarType, VariableGroup<V1, V2, ...>> {

SparseBlockMatrix A0; // sparse information matrix representing the prior uncertainty.
Vector b0; // dense column vector representing the mean of the prior.

GaussianPrior();

/// Adds variable to the gaussian prior with an initial uncertainty.
void addVariable(VariableKey<VariableType>& key, Eigen::Matrix<ScalarType, VariableType::dimension, VariableType::dimension>& informationMatrix);

/// Removes a variable from the
void removeVariable(VariableKey<VariableType>& removedKey);

// Updates the prior error term on manifold. A0 * (x + dx) = b0 => A0 * x = b0 - A0 * dx; 
void update(Vector dx);

};
```

### Marginalizer

The marginzalizer removes variables from the problem and approximates their effect by updating the gaussian prior
```cpp
class Marginalizer {

/// Marginalizes the variable requested using a set of linearized error terms.
void marginalizeVariable(GaussianPrior& prior, VariableKey<VariableType>& marginalizedKey, ErrorTermContainer<ErrorTermTypes...>& linearizedErrorTerms);

};
```

## Sparse Block Matrices
When, the dimensionality of our problem becomes large, it is not feasible to stores a NxN matrix where N is the dimensionality of our problem. To get around this problem, it is common to exploit the inherent sparsity of the problem. In SSEOptimizer, I want a sparse block matrix which is designed to work well with our variable container keys. 
### Sparse Block Matrix
```cpp
class SparseBlockMatrix<ScalarType, VariableGroup<V1, ...>> {

std::tuple<std::map<VariableKey<V1>, SparseBlockRow<ScalarType, V1::dimension, VariableGroup<V1, ...>>>, ...> rows;

// Returns the row map for the given key.
auto getMapForKey(VariableKey<V> key);

};
```

### Sparse Block Row
```cpp
class SparseBlockRow<ScalarType, RowDimension, VariableGroup<V1, ...>> {

template <size_t C>
using MatrixBlock = Eigen::Matrix<ScalarType, RowDimension, C>;

std::tuple<std::map<VariableKey<V1>, MatrixBlock<ScalarType, RowDimension, V1::dimension>>, ...> blocks;

// Returns a std map of blocks for the given variable type.
template <typename VariableType>
auto& getVariableMap();

// Computes the dot product of this row with a dense column vector.
// The variable container is used to determine the indices of each block.
void dot(VariableContainer<V1, ...>& variables, const Vector<ScalarType, N, M>& v, Eigen::Matrix<RowDimension, M>& result);

// Set all non zero elements to zero without deleting them
void setZero();

};
```

## Variable and ErrorTerm Requirements
### Optimizable Variable Requirements
```cpp
class OptimizableVariable<Scalar, VariableDimension> {

// The dimension if the minimal form of the variable.
static int Dimension = VariableDimension;

// Variable Data or Reference to data.
Variable;

// Generalized plus operation. Allows for on manifold optimization.
static void update(Eigen::Matrix<Scalar, Dimension, 1> delta);

};
```

### Error Term Requirements
```cpp
class ErrorTermBase<ScalarType, VariableGroup<V1, V2, ...>, ResidualDimension> {

// Stores the keys to the variables this error term is a function of.
tuple<VariableKey<V1>, VariableKey<V2>, ...> variableKeys;

// Linearized error term.
{
  bool linearized = false; // Flag which marks whether the error term was ever linearized.

  // The point at which the error term was last linearized.
  tuple<V1, V2, ...> linearizationPoint;

  // Jacobians of the error term w.r.t each variable at the linearization point.
  tuple<Eigen::Matrix<ScalarType, ResidualDimension, V1::dimension>, Eigen::Matrix<ScalarType, ResidualDimension, V2::dimension>, ...> jacobians;

  // Residual/Error at the linearization point.
  Eigen::Matrix<ScalarType, ResidualDimension, 1> residual;
}
};

class ErrorTerm : public ErrorTermBase<Scalar, VariableGroup<V1, V2, ...>, ResidualDimension> {

// Computes the error given the current state of the variables, optionally linearizes the error function.
void computeResidual(std::tuple<V1*, V2*, ...> variables, bool linearize = false);

};
```


