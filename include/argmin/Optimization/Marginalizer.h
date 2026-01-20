#pragma once

#include <cmath>
#include <numeric>
#include <type_traits>

#include "argmin/ErrorTerms/ErrorTermBase.h"
#include "argmin/Utilities/Logging.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Types/GaussianPrior.h"
#include "argmin/Containers/Key.h"
#include "argmin/Types/MetaHelpers.h"

namespace ArgMin {

template <typename... T>
class Marginalizer;

/**
 * The marginalizer is a class which is responsible for approximating the effect
 * of a variable's error terms on the rest of the variables.
 * This is done by approximating all error terms involving the error terms
 * with a single gaussian prior through the schur complement.
 */
template <typename ScalarType, typename... Variables, typename... ErrorTerms>
class Marginalizer<Scalar<ScalarType>, VariableGroup<Variables...>,
                   ErrorTermGroup<ErrorTerms...>> {
 public:
  template <typename RowVariable, typename ColumnVariable>
  using MatrixArray =
      SlotArray<Eigen::Matrix<ScalarType, RowVariable::dimension,
                              ColumnVariable::dimension>,
                VariableKey<ColumnVariable>>;

  template <typename RowVariable>
  using Row = std::tuple<MatrixArray<RowVariable, Variables>...>;

  Marginalizer() {}

  /// Marginalizes the variable requested using a set of linearized error terms.
  /// It will be assumed that the error term jacobians and residual are good
  /// approximations of the error term.
  /// After marginalization, any error terms containing the variable will be
  /// erased. After marginalization, the prior should be cleared of all
  /// marginalized variables. Otherwise it will contain excess information.
  ///
  /// Ignored variables, if already uncorrelated in the prior, will remain
  /// uncorrelated after marginalization. This is done by ignoring their off
  /// diagonal terms when computing the schur complement of the unmarginalized
  /// variables block.
  ///
  /// @param ignoredVariableTypes Jacobians with respect to these variables
  /// types will be ignored.
  /// @param marginalizedKey This variable will be marginalized into the prior.
  /// @param prior The marginalized information will be added to this prior.
  /// @param linearizedErrorTerms These are the error terms which will be used
  /// to compute the marginalized information.
  /// @param residualThreshold The threshold used to determine if an error term
  /// should be marginalized into the prior or simply culled.
  /// @return success flag signifying whether the marginalization was
  /// successful.
  template <typename VariableType, typename... IgnoredVariables>
  bool marginalizeVariable(
      VariableKey<VariableType> &marginalizedKey,
      GaussianPrior<Scalar<ScalarType>, VariableGroup<Variables...>> &prior,
      ErrorTermContainer<ErrorTerms...> &linearizedErrorTerms,
      VariableGroup<IgnoredVariables...> ignoredVariableTypes =
          VariableGroup<IgnoredVariables...>(),
      ScalarType residualThreshold = std::numeric_limits<ScalarType>::max()) {
    typedef typename std::remove_reference<decltype(
        marginalizedKey)>::type::variable_type MarginalizedVariable;

    LOG_TRACE("Marginalizing variable {} with index-generation {}-{}",
              typeid(VariableType).name(), marginalizedKey.index,
              marginalizedKey.generation);

    // Iterate through all error terms, and approximate the error terms which
    // are a function of the marginalized variable with a quadratic error term.
    // TODO I do not have to compute the notignored-ignored parts of the
    // quadratic cost.
    internal::static_for(
        linearizedErrorTerms.tupleOfErrorTermMaps,
        [&](auto i, auto &errorTermMap) {
          typedef
              typename std::tuple_element<i, std::tuple<ErrorTerms...>>::type
                  ThisErrorTerm;

          // A vector of error term keys which will be removed.
          std::vector<ErrorTermKey<ThisErrorTerm>> errorTermsToRemove;

          // Check if this error term is a function of the marginalized variable
          // type.
          if constexpr (internal::Is_in_tuple<
                            VariableKey<MarginalizedVariable>,
                            typename ThisErrorTerm::VariableKeys>::value) {
            // This error term type contains the same key type as the
            // marginalized key.
            for (auto errorTermIt = errorTermMap.begin();
                 errorTermIt != errorTermMap.end(); errorTermIt++) {
              auto &errorTerm = *(errorTermIt);

              // Runtime check if the marginalized variable key is in this error
              // term.
              bool errorTermContainsMarginalizedVariable = false;
              internal::static_for(
                  errorTerm.variableKeys, [&](auto i, auto &key) {
                    if (key == marginalizedKey) {
                      errorTermContainsMarginalizedVariable = true;
                    }
                  });

              // If this error term is a function of the marginalized variable,
              // add its approximation to the prior.
              if (errorTermContainsMarginalizedVariable) {
                // Check if the linearization is valid for this error term.
                if (errorTerm.linearizationValid) {
                  LOG_TRACE("Found valid error term associated to variable.");
                  // Check if the error term hass too high of an error.
                  if (errorTerm.residual.norm() < residualThreshold) {
                    LOG_TRACE(
                        "Found a valid and low error error term ({}) "
                        "associated to the variable being marginalized. The "
                        "error at the linearization point is: {}",
                        typeid(errorTerm).name(), errorTerm.residual.norm());
                    // Iterate through all independent variables.
                    internal::static_for(
                        errorTerm.variableKeys,
                        [&](auto i, auto &outerVariableKey) {
                          typedef typename std::remove_reference<decltype(
                              outerVariableKey)>::type OuterVariableKeyType;
                          typedef typename std::remove_reference<decltype(
                              errorTerm)>::type ErrorTermType;

                          // Cache the error transformation.
                          Eigen::Matrix<
                              ScalarType,
                              OuterVariableKeyType::variable_type::dimension,
                              ErrorTermType::residual_dimension>
                              rhoJtW = (std::get<i>(errorTerm.variableJacobians)
                                            .transpose() *
                                        errorTerm.information)
                                           .eval();

                          // Add to rhs.
                          prior.b0.getRowBlock(outerVariableKey) +=
                              rhoJtW * -errorTerm.residual;

                          internal::static_for(
                              errorTerm.variableKeys,
                              [&](auto j, auto &innerVariableKey) {
                                // If this variable key points to a variable
                                // type which should be ignored, skip.
                                // if constexpr (!internal::Is_in_tuple<typename
                                // decltype(innerVariableKey)::variable_type,
                                // std::tuple<IgnoredVariables>>::value)
                                //{
                                // Extract the variable types from the keys.
                                // These keys must be of type
                                // VariableKey<VariableType>.
                                typedef typename std::remove_reference<decltype(
                                    innerVariableKey)>::type
                                    InnerVariableKeyType;

                                // Compute pJtJ and pJte for this error term.
                                // Add to lhs.
                                prior.A0.getBlock(outerVariableKey,
                                                  innerVariableKey) +=
                                    rhoJtW *
                                    std::get<j>(errorTerm.variableJacobians);
                                //}
                              });
                        });
                  }
                } else {
                  LOG_WARN("linearization invalid for error term. Skipping");
                }

                // Add this error term to the deletion queue.
                errorTermsToRemove.push_back(errorTermMap.getKeyFromDataIndex(
                    errorTermIt - errorTermMap.begin()));
              }
            }
          }

          // Erase all the approximated error terms.
          for (const auto &key : errorTermsToRemove) {
            linearizedErrorTerms.erase(key);
          }
        });

    // Compute the remaining part of the schur complement of the remaining
    // variable block. this is the A0_remain - B' * inv(A) * B

    // Get the marginalized row. This assumes that the row exists.
    CHECK(prior.A0.template getRowMap<MarginalizedVariable>().count(
              marginalizedKey) == 1,
          "There must be a row for a variable to be marginalized.");
    auto &marginalizedRow =
        prior.A0.template getRowMap<MarginalizedVariable>().at(marginalizedKey);
    std::tuple<Variables *...> variableTuple;

    Eigen::Matrix<ScalarType, MarginalizedVariable::dimension,
                  MarginalizedVariable::dimension> &A_marg =
        prior.A0.getBlock(marginalizedKey, marginalizedKey);

    Eigen::Matrix<ScalarType, MarginalizedVariable::dimension,
                  MarginalizedVariable::dimension>
        A_margInv = A_marg.inverse();
    // Verify that the matrix inverse was valid.
    if (std::isnan(A_margInv.sum())) {
      LOG_ERROR(
          "Marginalization failed due to failed inversion of the marginalized "
          "block. Returning early.");
      return false;
    }

    internal::static_for(variableTuple, [&](auto i, auto &doNotUseMe) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          OuterVariable;

      // Don't use ignored variables.
      if constexpr (!internal::Is_in_tuple<
                        OuterVariable,
                        std::tuple<IgnoredVariables...>>::value) {
        auto &outerVariableMap =
            marginalizedRow.template getVariableMap<OuterVariable>();

        for (auto &outerPair : outerVariableMap) {
          // Precompute B' * inv(A)
          Eigen::Matrix<ScalarType, OuterVariable::dimension,
                        MarginalizedVariable::dimension>
              bTAinv = (outerPair.second.transpose() * A_margInv).eval();

          internal::static_for(variableTuple, [&](auto j,
                                                  auto &doNotUseMeAlso) {
            typedef
                typename std::tuple_element<j, std::tuple<Variables...>>::type
                    InnerVariable;

            // Don't use ignored variables.
            if constexpr (!internal::Is_in_tuple<
                              InnerVariable,
                              std::tuple<IgnoredVariables...>>::value) {
              auto &innerVariableMap =
                  marginalizedRow.template getVariableMap<InnerVariable>();

              for (auto &innerPair : innerVariableMap) {
                // If the inner variable and outer variable are both the same
                // type as the marginalized variable Check if they are the
                // marginalized variable.
                if constexpr (std::is_same<OuterVariable,
                                           MarginalizedVariable>::value &&
                              std::is_same<InnerVariable,
                                           MarginalizedVariable>::value) {
                  if (marginalizedKey == innerPair.first &&
                      marginalizedKey == outerPair.first) {
                    // Skip this block.
                    continue;
                  }

                  // Subtract the projected information from the prior.
                  prior.A0.getBlock(outerPair.first, innerPair.first) -=
                      bTAinv * innerPair.second;
                }
              }
            }
          });
        }
      }
    });

    // Successful marginalization
    return true;
  }
};  // namespace ArgMin

}  // namespace ArgMin
