#pragma once

#include <map>
#include "argmin/Types/MetaHelpers.h"
#include "argmin/Types/SparseBlockRow.h"

namespace ArgMin
{

template <typename...>
class SparseBlockMatrix;

template <typename ScalarType, typename... Variables>
class SparseBlockMatrix<Scalar<ScalarType>, VariableGroup<Variables...>>
{
public:
    template <typename VariableType>
    using Row = SparseBlockRow<Scalar<ScalarType>, Dimension<VariableType::dimension>, VariableGroup<Variables...>>;

    template <typename VariableType>
    using RowMap = std::map<VariableKey<VariableType>, Row<VariableType>>;

    SparseBlockMatrix() {}

    /// Gets the map of rows for a given variable type
    template <typename VariableType>
    RowMap<VariableType>& getRowMap()
    {
        return std::get<RowMap<VariableType>>(tupleOfRowMaps);
    }

    /// Inserts a row into the matrix and return its reference.
    template <typename VariableType>
    auto& addRowIfItDoesNotExist(VariableKey<VariableType> key)
    {
        // Only insert if the key is not in the map.
        auto rowIt = getRowMap<VariableType>().find(key);

        if (rowIt == getRowMap<VariableType>().end())
        {
            std::tie(rowIt, std::ignore) = getRowMap<VariableType>().insert(std::make_pair(key, Row<VariableType>()));
        }

        return rowIt->second;
    }

    /// Gets a block matrix reference from the matrix. If the block does not exist, it is inserted and set to zero.
    /// @param rowKey key pointing to the row.
    /// @param columnKey key pointing to the column.
    template <typename RowType, typename ColType>
    Eigen::Matrix<ScalarType, RowType::dimension, ColType::dimension>& getBlock(VariableKey<RowType> rowKey, VariableKey<ColType> colKey)
    {
        // Add a row to the sbm.
        auto& row = addRowIfItDoesNotExist(rowKey);

        auto result = row.template getVariableMap<ColType>().insert(std::make_pair(colKey, Eigen::Matrix<ScalarType, RowType::dimension, ColType::dimension>::Zero()));

        return result.first->second;
    }

    /// Inserts or assigns a block at the given keys.
    template <typename RowType, typename ColType>
    void setBlock(VariableKey<RowType> rowKey, VariableKey<ColType> colKey, const Eigen::Matrix<ScalarType, RowType::dimension, ColType::dimension>& blockMatrix)
    {
        // Add a row to the sbm.
        auto& row = addRowIfItDoesNotExist(rowKey);

        // insert or assign the block
        row.template getVariableMap<ColType>().insert_or_assign(colKey, blockMatrix);
    }

    /// Removes a block using the keys. Essentially, sets the block to zero.
    template <typename RowType, typename ColType>
    void removeBlock(VariableKey<RowType> rowKey, VariableKey<ColType> colKey)
    {
        // get the row.
        auto row = getRowMap<RowType>().find(rowKey);

        // If the row exists remove the column.
        if (row != getRowMap<RowType>().end())
        {
            row->second.template getVariableMap<ColType>().erase(colKey);
        }
        
    }

    

    /// Computes the dot product of this Matrix with a dense column vector.
    /// The variable container is used to determine the indices of each block.
    /// a.k.a. A * v = result
    template <int DenseMatrixColumns>
    void dot(VariableContainer<Variables...> &variableOrder, const Eigen::Matrix<ScalarType, Eigen::Dynamic, DenseMatrixColumns> &v, Eigen::Matrix<ScalarType, Eigen::Dynamic, DenseMatrixColumns>& result)
    {
        // Not that efficient, but the result vector must be zero by default
        result.setZero();

        internal::static_for(tupleOfRowMaps, [&](auto i, auto &rowMap) {
            auto &map = variableOrder.template getVariableMap<typename std::tuple_element<i, std::tuple<Variables...>>::type>();

            assert(rowMap.size() <= map.size()); // There should be no more variables in the SBM than are in the variable container.

            if (map.size() > 0)
            {
                auto firstVariableKey = map.getKeyFromDataIndex(0);
                // Precompute the offset index for the current variable type.
                auto startingIndex = variableOrder.template variableIndex(firstVariableKey);

                Eigen::Matrix<ScalarType, std::tuple_element<i, std::tuple<Variables...>>::type::dimension, DenseMatrixColumns> blockResult;

                for (auto &pair : rowMap)
                {
                    auto variableIterator = map.at(pair.first);
                    if (variableIterator != map.end())
                    {
                        auto index = startingIndex + (variableIterator - map.begin()) * std::tuple_element<i, std::tuple<Variables...>>::type::dimension;

                        pair.second.dot(variableOrder, v, blockResult);
                        result.block(index, 0, std::tuple_element<i, std::tuple<Variables...>>::type::dimension, DenseMatrixColumns) = blockResult;
                    }
                    else {
                        assert(false && "variable in SBM not in variable container!");
                    }
                }
            }
        });
    }

    private:

    std::tuple<RowMap<Variables>...> tupleOfRowMaps;
};

} // namespace ArgMin