#pragma once

#include "argmin/Containers/SlotMap.h"

namespace ArgMin {

template <typename T>
class VariableKey : public SlotMapKeyBase {
 public:
  // This can be used to extract the type of the key variable at compile time.
  typedef T variable_type;
  // Slot Map key (index, generation).
  bool operator<(const VariableKey<T> &other) const {
    if (this->index == other.index) {
      return this->generation < other.generation;
    } else {
      return this->index < other.index;
    }
  }

  /// Compares two keys by their index, but not generation.
  bool operator==(const VariableKey<T> &other) const {
    return this->index == other.index && this->generation == other.generation;
  }

  /// Compares two keys by their index, but not generation.
  template <typename Other>
  bool operator==(const VariableKey<Other> &other) const {
    return false;
  }
};

template <typename T>
class ErrorTermKey : public SlotMapKeyBase {
 public:
  typedef T errorterm_type;
  // Slot Map key (index, generation).
  bool operator<(const ErrorTermKey<T> &other) const {
    if (this->index == other.index) {
      return this->generation < other.generation;
    } else {
      return this->index < other.index;
    }
  }
};

}  // namespace ArgMin
