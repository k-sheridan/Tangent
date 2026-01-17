#pragma once

#include "argmin/MetaHelpers.h"
#include "argmin/Key.h"
#include "argmin/Containers.h"

namespace ArgMin
{

template <typename...>
class SSEOptimizer;

template <typename ScalarType, typename Solver, typename Prior, typename Marginalizer, typename... Variables, typename... ErrorTerms>
class SSEOptimizer<Scalar<ScalarType>, Solver, Prior, Marginalizer, VariableGroup<Variables...>, ErrorTermGroup<ErrorTerms...>>
{
public:
    /// Inserts a variable into the optimizer.
    template <typename VariableType>
    VariableKey<VariableType> addVariable(VariableType &optimizableVariable)
    {
        return variables.template getVariableMap<VariableType>().insert(optimizableVariable);
    }

    // Updates the marginal information of the given variable.
    template <typename VariableType>
    void setVariablePrior(VariableKey<VariableType> &variableKey, Eigen::Matrix<ScalarType, VariableType::dimension, VariableType::dimension> &covariance)
    {
        assert(false);
    }

    /// Removes a variable and it's associated error terms, while approximating the 
    /// information they provided to the remaining variables with a gaussian prior.
    template <typename VariableType>
    void marginalizeVariable(VariableKey<VariableType>& key)
    {
        assert(false);
    }

    /// Removes a variable from the problem without marginalizing it.
    template <typename VariableType>
    void removeVariable(VariableKey<VariableType>& key)
    {
        variables.template getVariableMap<VariableType>().erase(key);

        
    }

    /// Adds error term to the problem.
    template <typename ErrorTermType>
    ErrorTermKey<ErrorTermType> addErrorTerm(ErrorTermType& errorTerm)
    {
        assert(false);
    }

    /// Removes error term from the problem.
    template <typename ErrorTermType>
    void removeErrorTerm(ErrorTermKey<ErrorTermType>& errorTermKey)
    {
        assert(false);
    }

    /// Iteratively refines the variables until the SSE of the error terms is minimized.
    void optimize()
    {
        assert(false);
    }

    /// Variable container.
    VariableContainer<Variables...> variables;

    /// Error term container.
    ErrorTermContainer<ErrorTerms...> errorTerms;

    /// Prior
    Prior prior;

    /// Marginalizer
    Marginalizer marginalizer;

    /// Solver
    Solver solver;


};

} //namespace ArgMin