#pragma once

#include <Eigen/Core>

#include "argmin/Types/MetaHelpers.h"
#include "argmin/Types/BlockVector.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Containers/Key.h"
#include "argmin/Types/SparseBlockMatrix.h"
#include "argmin/Types/SparseBlockRow.h"
#include "argmin/Utilities/Logging.h"

namespace ArgMin {

template <typename... T>
class GaussianPrior;

template <typename ScalarType, typename... Variables>
class GaussianPrior<Scalar<ScalarType>, VariableGroup<Variables...>> {
 public:
  static constexpr ScalarType DefaultInverseVariance = 1e-24;

  using SBM =
      SparseBlockMatrix<Scalar<ScalarType>, VariableGroup<Variables...>>;

  using BV = BlockVector<Scalar<ScalarType>, Dimension<1>,
                         VariableGroup<Variables...>>;

  SBM A0;  // Sparse Block Information Matrix.

  BV b0;  // dense mean vector of the gaussian prior.

  Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>
      temporaryVector;  // Preallocated vector.

  GaussianPrior() {}

  /// Adds variable to the gaussian prior with an initial uncertainty.
  template <typename VariableType>
  void addVariable(VariableKey<VariableType>& key,
                   Eigen::Matrix<ScalarType, VariableType::dimension,
                                 VariableType::dimension>
                       informationMatrix =
                           Eigen::Matrix<ScalarType, VariableType::dimension,
                                         1>::Constant(DefaultInverseVariance)
                               .asDiagonal()) {
    // insert the information block
    A0.setBlock(key, key, informationMatrix);

    // Insert a zero mean.
    Eigen::Matrix<ScalarType, VariableType::dimension, 1> zeroVec =
        Eigen::Matrix<ScalarType, VariableType::dimension, 1>::Zero();
    b0.addRowBlock(key, zeroVec);
  }

  /// Remove all variables from the gaussian prior which are not part of the
  /// given set of variables. This operation has a complexity of N^2 where N is
  /// the number of elements in a row. This is the worst case and will typically
  /// be much lower since the prior is typically very sparse.
  void removeUnsedVariables(
      VariableContainer<Variables...>& variableContainer) {
    // This tuple is used to iterate over all variable types.
    std::tuple<ArgMin::TypedSlotMapKey<Variables>...> tupleOfVariables;

    // Iterate through all sparse block rows.
    internal::static_for(tupleOfVariables, [&](auto i, auto& do_not_use_me) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          RowVariable;
      auto& rowMap = A0.template getRowMap<RowVariable>();

      std::vector<VariableKey<RowVariable>> rowsToRemove;

      for (auto rowIt = rowMap.begin(); rowIt != rowMap.end(); rowIt++) {
        // Get the key for this row.
        const VariableKey<RowVariable>& rowKey = rowIt->first;
        // Check if the key exists in the variable set.
        if (variableContainer.variableExists(rowKey)) {
          // Search the sparse block row for invalid elements.
          internal::static_for(tupleOfVariables, [&](auto j,
                                                     auto& do_not_use_me_too) {
            typedef
                typename std::tuple_element<j, std::tuple<Variables...>>::type
                    ColumnVariable;
            auto& columnMap =
                rowIt->second.template getVariableMap<ColumnVariable>();

            std::vector<VariableKey<ColumnVariable>> colsToRemove;

            for (auto colIt = columnMap.begin(); colIt != columnMap.end();
                 colIt++) {
              // Get the key for this block matrix.
              const VariableKey<ColumnVariable>& colKey = colIt->first;

              // Check if this key exists.
              if (!variableContainer.variableExists(colKey)) {
                colsToRemove.push_back(colKey);
              }
            }

            // Remove the columns.
            for (const auto& key : colsToRemove) {
              columnMap.erase(key);
            }
          });
        } else {
          // Delete this row.
          rowsToRemove.push_back(rowKey);

          // Delete the row's corresponding b0 block.
          b0.removeRowBlock(rowKey);
        }
      }
      // Remove the rows.
      for (const auto& key : rowsToRemove) {
        LOG_TRACE("Removing ({}) key with index-generation {}-{} from prior",
                  typeid(key).name(), key.index, key.generation);
        rowMap.erase(key);
      }
    });
  }

  /// Updates the prior error term on manifold. A0 * (x + dx) = b0 => A0 * x =
  /// b0 - A0 * dx; Assumes that the dx vector has the same variable order as
  /// the variable container.
  void update(VariableContainer<Variables...>& variableOrder,
              const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& dx) {
    // Uneccesarily expensive. This can be packaged directly into the dot
    // function. This is ran once per iteration.
    size_t problemSize = variableOrder.totalDimensions();
    assert(dx.rows() == problemSize);  // ensure that dx is consistent with the
                                       // current variable set.

    if (problemSize > temporaryVector.rows()) {
      temporaryVector.resize(problemSize, Eigen::NoChange);
    }

    // Compute the perturbation.
    A0.dot(variableOrder, dx, temporaryVector);

    // Move the mean.
    b0.subtractVector(variableOrder, temporaryVector);
  }

  /// Updates the prior with a block vector.
  /// The block vector must contain at least a subset of the variables contained
  /// in the b0 vector. This is the most efficient variant of this operation and
  /// is agnostic to ordering.
  void update(BlockVector<Scalar<ScalarType>, Dimension<1>,
                          VariableGroup<Variables...>>& dx) {
    // This tuple is used to iterate over all variable types.
    std::tuple<Variables*...> tupleOfVariables;

    // Compute b0 = b0 - A0*dx
    internal::static_for(tupleOfVariables, [&](auto i, auto& do_not_use_me) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          RowVariable;

      auto& rowMap = A0.template getRowMap<RowVariable>();

      // Stores the dot product of a row and dx vector.
      Eigen::Matrix<ScalarType, RowVariable::dimension, 1> temporary;

      // Iterate over all rows of the SBM.
      for (auto& keyRowPair : rowMap) {
        const VariableKey<RowVariable>& key = keyRowPair.first;
        auto& sparseBlockRow = keyRowPair.second;

        // Compute the dot product of the sparse block row with the dx vector.
        sparseBlockRow.dot(dx, temporary);

        // The b0 block must exist for this key.
        assert(b0.blockExists(key));

        auto& bBlock = b0.getRowBlock(key);

        bBlock.noalias() -= temporary;
      }
    });
  }
};

}  // namespace ArgMin
