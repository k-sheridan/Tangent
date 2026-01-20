#pragma once

#include "argmin/Types/MetaHelpers.h"
#include "argmin/Containers/SlotMap.h"
#include "argmin/Containers/Key.h"

namespace ArgMin
{

template <typename... Variables>
class VariableContainer
{
public:
   template <typename VariableType>
   using VariableMap = SlotMap<VariableType, VariableKey<VariableType>>;

   /// Gets the map container for a variable type.
   template <typename VariableType>
   VariableMap<VariableType> &getVariableMap()
   {
      return std::get<VariableMap<VariableType>>(tupleOfVariableMaps);
   }
   /// Inserts a variable into the container.
   /// @return The key to the variable is returned.
   template <typename VariableType>
   VariableKey<VariableType> insert(const VariableType& value)
   {
      return getVariableMap<VariableType>().insert(value);
   }

   /// Gets the variable by its key.
   /// @return A reference to the value in the container.
   template <typename VariableType>
   VariableType& at(const VariableKey<VariableType>& key)
   {
      return *(getVariableMap<VariableType>().at(key));
   }

   /// Erase a variable into the container.
   template <typename VariableType>
   void erase(VariableKey<VariableType>& key)
   {
      getVariableMap<VariableType>().erase(key);
   }

   /// Checks if a key is a valid variable key.
   template <typename VariableType>
   bool variableExists(VariableKey<VariableType> key)
   {
      return getVariableMap<VariableType>().at(key) != getVariableMap<VariableType>().end();
   }

   /// Gets the index of the first scalar of the given variable.
   /// This is used to build and operate on a matrix.
   /// O(N) with number of variable types.
   template <typename VariableType>
   size_t variableIndex(VariableKey<VariableType> &key)
   {
      size_t variableIndex = 0;
      static const int tupleIndexOfVariable = internal::Index<VariableType, std::tuple<Variables...>>::value;

      internal::static_for(tupleOfVariableMaps, [&](auto i, auto &v) {
         if constexpr (i < tupleIndexOfVariable)
         {
            variableIndex += v.size() * std::tuple_element<i, std::tuple<Variables...>>::type::dimension;
         }

         if constexpr (i == tupleIndexOfVariable)
         {
            auto it = v.at(key);
            assert(it != v.end());

            variableIndex += (it - v.begin()) * std::tuple_element<i, std::tuple<Variables...>>::type::dimension;
         }

      });

      return variableIndex;
   }

   /// Computes the total dimensionality of the variables stored in this container.
   size_t totalDimensions()
   {
      size_t dims = 0;
      internal::static_for(tupleOfVariableMaps, [&](auto i, auto &v) { dims += v.size() * std::tuple_element<i, std::tuple<Variables...>>::type::dimension; });
      return dims;
   }

   // Stores the variables.
   std::tuple<VariableMap<Variables>...> tupleOfVariableMaps;
};

/**
 * Container for all error terms.
 */
template <typename... ErrorTerms>
class ErrorTermContainer
{
public:
   template <typename ErrorTermType>
   using ErrorTermMap = SlotMap<ErrorTermType, ErrorTermKey<ErrorTermType>>;

   /// Gets the map container for a variable type.
   template <typename ErrorTermType>
   ErrorTermMap<ErrorTermType> &getErrorTermMap()
   {
      return std::get<ErrorTermMap<ErrorTermType>>(tupleOfErrorTermMaps);
   }

   /// Gets the variable by its key.
   /// @return A reference to the value in the container.
   template <typename ErrorTermType>
   ErrorTermType& at(const ErrorTermKey<ErrorTermType>& key)
   {
      return *(getErrorTermMap<ErrorTermType>().at(key));
   }

   /// Erases an error term from the container.
   /// @param key error term key which will be removed.
   template <typename ErrorTermType>
   void erase(const ErrorTermKey<ErrorTermType>& key)
   {
      (getErrorTermMap<ErrorTermType>().erase(key));
   }

   /// Inserts an error term into the container.
   /// @return The key refering to the error term in the container.
   template <typename ErrorTermType>
   ErrorTermKey<ErrorTermType> insert(const ErrorTermType& errorTerm)
   {
      return getErrorTermMap<ErrorTermType>().insert(errorTerm);
   }

   /// Clears the error term container of all error terms.
   void clear() {
      internal::static_for((tupleOfErrorTermMaps), [&](auto i, auto &variableMap) {
         variableMap.clear();
      });
   }

   /// Stores the error terms.
   std::tuple<ErrorTermMap<ErrorTerms>...> tupleOfErrorTermMaps;
};

} // namespace ArgMin
