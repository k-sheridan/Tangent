#pragma once

#include "argmin/Containers/SlotArray.h"
#include "argmin/Containers/Key.h"
#include "argmin/Types/MetaHelpers.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include <Eigen/Core>
#include <tuple>

namespace ArgMin {

template <typename... T>
class BlockVector;

/**
 * Unlike the SparseBlockRow, the BlockVector is assumed to be dense and can then 
 * achieve constant time insert, lookup, and erase via the slot map.
 * This class uses a slightly different variant of the slot map. One which
 * allows us to choose a key value pair.
 */
template <typename ScalarType, int ColumnDimension, typename... Variables>
class BlockVector<Scalar<ScalarType>, Dimension<ColumnDimension>, VariableGroup<Variables...>> {
public:
    template <typename VariableType>
    using MatrixBlock = Eigen::Matrix<ScalarType, VariableType::dimension, ColumnDimension>;
    template <typename VariableType>
    using RowMap = SlotArray<MatrixBlock<VariableType>, VariableKey<VariableType>>;

    BlockVector() {}
    
    /**
     * Gets a reference to the internal row map for a given variable type.
     */
    template <typename VariableType>
    RowMap<VariableType>& getRowMap()
    {
        return std::get<RowMap<VariableType>>(tupleOfRowMaps);
    }

    /**
     * Gets a const reference to the internal row map for a given variable type.
     */
    template <typename VariableType>
    const RowMap<VariableType>& getRowMap() const
    {
        return std::get<RowMap<VariableType>>(tupleOfRowMaps);
    }

    /// Inserts a key value pair into the slot map.
    /// This should never need overwrite an element. 
    template <typename VariableType>
    typename RowMap<VariableType>::InsertResult addRowBlock(VariableKey<VariableType> key, MatrixBlock<VariableType>& value)
    {
        
        return std::get<RowMap<VariableType>>(tupleOfRowMaps).insert(key, value);
        
    }

    /// Removes the row block.
    template <typename VariableType>
    void removeRowBlock(VariableKey<VariableType> key)
    {

        // Erase the slot map element at the internal key.
        std::get<RowMap<VariableType>>(tupleOfRowMaps).erase(key);

    }

    /// Gets a reference to the block matrix at the key.
    template <typename VariableType>
    MatrixBlock<VariableType>& getRowBlock(VariableKey<VariableType> key)
    {
        assert(blockExists(key));
        return *(std::get<RowMap<VariableType>>(tupleOfRowMaps).at(key));
    }

    /// Evaluates if the key exists in the block vector.
    template <typename VariableType>
    bool blockExists(VariableKey<VariableType> key)
    {
        return std::get<RowMap<VariableType>>(tupleOfRowMaps).at(key) != std::get<RowMap<VariableType>>(tupleOfRowMaps).end();
    }

    /// Computes BlockVector -= v.
    void subtractVector(VariableContainer<Variables...> &variableOrder, const Eigen::Matrix<ScalarType, Eigen::Dynamic, ColumnDimension> &v)
    {
        internal::static_for(variableOrder.tupleOfVariableMaps, [&](auto i, auto &variableMap){
            auto &map = variableOrder.template getVariableMap<typename std::tuple_element<i, std::tuple<Variables...>>::type>();
            constexpr int variableDimension = std::tuple_element<i, std::tuple<Variables...>>::type::dimension;

            MatrixBlock<typename std::tuple_element<i, std::tuple<Variables...>>::type> zero = MatrixBlock<typename std::tuple_element<i, std::tuple<Variables...>>::type>::Zero();

            if (map.size() > 0)
            {
                auto firstVariableKey = map.getKeyFromDataIndex(0);
                // Precompute the offset index for the current variable type.
                auto startingIndex = variableOrder.template variableIndex(firstVariableKey);

                for (auto variableIterator = map.begin(); variableIterator != map.end(); variableIterator++)
                {
                    auto key = map.getKeyFromDataIndex(variableIterator - map.begin());
                    int index = startingIndex + (variableIterator - map.begin()) * variableDimension;

                    const auto& rhs = v.block(index, 0, variableDimension, 1);

                    if (blockExists(key))
                    {
                        auto& rowBlock = getRowBlock(key);

                        rowBlock -= rhs;
                    }
                    else
                    {
                        // Add the row.
                        addRowBlock(key, zero);
                        getRowBlock(key) -= rhs;
                    }

                }
            }
        });
    }

private:

std::tuple<RowMap<Variables>...> tupleOfRowMaps;

};

} // namespace ArgMin
