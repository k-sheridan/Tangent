#pragma once

#include <Eigen/Core>
#include "argmin/Containers/Key.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Types/MetaHelpers.h"

namespace ArgMin
{

template <typename...>
class ErrorTermBase;

/**
 * This serves as the base of a simple error term with only a handful of independent variables.
 * This base stores the linearized error term and the pointers to to extract the variables.
 * 
 * The derived error term may contain more information than this base class, but it MUST contain
 * a function: 
 * 
 * void evaluate(VariableContainer<Variables...>& variables, bool relinearize);
 * 
 * This function must compute the residual given the variable states, and will compute the jacobians w.r.t to the variables
 * if relinearize is true. Check the linearizationValid flag before using the error term.
 */
template <int ResidualDimension, typename ScalarType, typename... IndependentVariables>
class ErrorTermBase<Scalar<ScalarType>, Dimension<ResidualDimension>, VariableGroup<IndependentVariables...>>
{
public:

    using VariablePointers = std::tuple<IndependentVariables*...>;
    using VariableKeys = std::tuple<VariableKey<IndependentVariables>...>;
    using VariableJacobians = std::tuple<Eigen::Matrix<ScalarType, ResidualDimension, IndependentVariables::dimension>...>;

    // The precision of this error term.
    typedef ScalarType scalar_type;

    /// Compile time acces to the error term's dimension.
    static const int residual_dimension = ResidualDimension;

    /// These jacobians are from the most recent linearization.
    VariableJacobians variableJacobians;
    /// These are the keys used to access the variables over time.
    /// These keys can only be invalidated is the variable is removed or overwritten.
    VariableKeys variableKeys;
    /// These pointers are used to avoid the indirection of the slotmap.
    /// These pointers are invalidated every time a key is added or removed from the slot map.
    VariablePointers variablePointers;

    /// This is the most recent residual computed for the error term.
    Eigen::Matrix<ScalarType, ResidualDimension, 1> residual;

    /// This is the information matrix for this error term.
    Eigen::Matrix<ScalarType, ResidualDimension, ResidualDimension> information;

    /// This flag is used to let the user know if the error term was linearized successfully.
    bool linearizationValid = false;

    /// Extracts the most recent pointer to each of the variables using their key and updates them.
    template <typename... Variables>
    void updateVariablePointers(VariableContainer<Variables...> &variableContainer)
    {
        internal::static_for(variableKeys, [&](auto i, auto &variableKey) {
            auto& variableMap = variableContainer.template getVariableMap<typename std::tuple_element<i, std::tuple<IndependentVariables...>>::type>();
            auto variableIterator = variableMap.at(variableKey);

            assert(variableIterator != variableMap.end());

            std::get<i>(variablePointers) = &(*(variableIterator));
        });
    }

    /// Verifies that the pointers stored are equal to the true variable location.
    template <typename... Variables>
    bool checkVariablePointerConsistency(VariableContainer<Variables...> &variableContainer)
    {
        bool flag = true;
        internal::static_for(variableKeys, [&](auto i, auto &variableKey) {
            auto& variableMap = variableContainer.template getVariableMap<typename std::tuple_element<i, std::tuple<IndependentVariables...>>::type>();
            auto variableIterator = variableMap.at(variableKey);

            assert(variableIterator != variableMap.end());

            // The pointers should match.
            if (std::get<i>(variablePointers) != &(*(variableIterator)))
            {
                flag = false;
            }
        });

        return flag;
    }
};

} // namespace ArgMin