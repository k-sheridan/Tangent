#pragma once

#include <map>
#include <tuple>
#include <Eigen/Core>
#include "argmin/Types/MetaHelpers.h"
#include "argmin/Containers/SlotMap.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Types/BlockVector.h"

namespace ArgMin
{

template <typename...>
class SparseBlockRow;

template <typename ScalarType, int RowDimension, typename... Variables>
class SparseBlockRow<Scalar<ScalarType>, Dimension<RowDimension>, VariableGroup<Variables...>>
{
public:
    template <typename VariableType>
    using MatrixBlock = Eigen::Matrix<ScalarType, RowDimension, VariableType::dimension>;

    template <typename VariableType>
    using VariableColumns = std::map<VariableKey<VariableType>, MatrixBlock<VariableType>>;

    SparseBlockRow() {}

    /// Returns a std map of blocks for the given variable type.
    template <typename VariableType>
    VariableColumns<VariableType> &getVariableMap()
    {
        return std::get<VariableColumns<VariableType>>(columns);
    }

    /// Computes the dot product of this row with a dense column vector.
    /// The variable container is used to determine the indices of each block.
    /// a.k.a. Row * v = result
    template <int DenseMatrixColumns>
    void dot(VariableContainer<Variables...> &variableOrder, const Eigen::Matrix<ScalarType, Eigen::Dynamic, DenseMatrixColumns> &v, Eigen::Matrix<ScalarType, RowDimension, DenseMatrixColumns>& result)
    {
        result = Eigen::Matrix<ScalarType, RowDimension, DenseMatrixColumns>::Zero();

        internal::static_for(columns, [&](auto i, auto &matrixMap) {
            auto &map = variableOrder.template getVariableMap<typename std::tuple_element<i, std::tuple<Variables...>>::type>();

            if (map.size() > 0)
            {
                auto firstVariableKey = map.getKeyFromDataIndex(0);
                // Precompute the offset index for the current variable type.
                auto startingIndex = variableOrder.template variableIndex(firstVariableKey);

                for (auto &pair : matrixMap)
                {
                    auto variableIterator = map.at(pair.first);
                    if (variableIterator != map.end())
                    {
                        auto index = startingIndex + (variableIterator - map.begin()) * std::tuple_element<i, std::tuple<Variables...>>::type::dimension;

                        result += pair.second * v.block(index, 0, std::tuple_element<i, std::tuple<Variables...>>::type::dimension, DenseMatrixColumns);
                    }
                }
            }
        });
    }

    /// Block vector variant of the sparse dot product.
    /// If the the block vector does not contain an element in the sparse block row, it is assumed to be 0.
    /// a.k.a. Row * v = result
    template <int DenseMatrixColumns>
    void dot(BlockVector<Scalar<ScalarType>, Dimension<DenseMatrixColumns>, VariableGroup<Variables...>> &v, Eigen::Matrix<ScalarType, RowDimension, DenseMatrixColumns>& result)
    {
        result = Eigen::Matrix<ScalarType, RowDimension, DenseMatrixColumns>::Zero();

        internal::static_for(columns, [&](auto i, auto &matrixMap) {
            typedef typename std::tuple_element<i, std::tuple<Variables...>>::type ThisVariable;
            for (const auto& keyBlockPair : matrixMap)
            {
                const VariableKey<ThisVariable>& key = keyBlockPair.first;

                // If the block does not exist, don't dot this element.
                if (v.blockExists(key)) 
                {
                    const Eigen::Matrix<ScalarType, ThisVariable::dimension, DenseMatrixColumns>& vBlock = v.getRowBlock(key);

                    result.noalias() += keyBlockPair.second * vBlock;
                }
            }
        });

    }

    /// Sets all current non zero blocks to zero.
    void setZero()
    {
        internal::static_for(columns, [](auto i, auto& v) {
            for (auto& pair : v)
            {
                pair.second.setZero();
            }
        });
    }

private:
    std::tuple<VariableColumns<Variables>...> columns; // Main storage for the matrix blocks in the row.
};

} // namespace ArgMin