#pragma once

#include <cassert>
#include <vector>

namespace ArgMin {

/**
 * The slot array is basically a vector which uses keys and 1
 * layer of indirection to allow for constant time insert, erase,
 * and access. Additionally, this implementation allows for fast iteration.
 */
template <typename DataType, typename KeyType>
class SlotArray {
  static_assert(std::is_integral<typename KeyType::index_type>::value);
  static_assert(std::is_integral<typename KeyType::generation_type>::value);

  struct Slot {
    /// The current generation of this slot.
    typename KeyType::generation_type generation;
    size_t dataIndex;  // index of the data in the data array.
    bool free = true;
  };

  std::vector<Slot> slots;  // Set of keys currently in use.

  std::vector<DataType> data;  // vector containing all data.
  std::vector<size_t>
      dataToSlotIndex;  // vector which points data to its slot index.

 public:
  enum InsertResult { SUCCESS_NO_OVERWRITE, SUCCESS_OVERWRITE, FAILURE };

  SlotArray() {}

  /**
   * O(1)
   * The insert result lets the user know if an element was overwritten, or if a
   * failure occured.
   */
  InsertResult insert(KeyType key, DataType value) {
    const size_t slotIndex = key.index;

    // Insert a new slot if necessary.
    if (slotIndex >= slots.size()) {
      slots.resize(slotIndex + 1);
    }

    // get slot reference
    Slot &slot = slots.at(slotIndex);

    // Update the slot generation.
    slot.generation = key.generation;

    // Is the slot free?
    if (slot.free) {
      // set up slot
      // set slot to not free
      slot.free = false;
      slot.dataIndex = data.size();

      // push a new data member to the back of the data arrays.
      data.push_back(std::move(value));
      dataToSlotIndex.push_back(slotIndex);

      return InsertResult::SUCCESS_NO_OVERWRITE;

    } else {
      // Overwrite the slot.
      data.at(slot.dataIndex) = value;
      assert(dataToSlotIndex.at(slot.dataIndex) == slotIndex);

      return InsertResult::SUCCESS_OVERWRITE;
    }
  }

  /**
   * O(1)
   * This variant of insert requires that the data type has a
   * default constructor.
   *
   * If, data already exists at this key the data is returned by reference.
   * If, the key does not exist, data is emplaced and returned.
   */
  DataType &insert(KeyType key) {
    const size_t slotIndex = key.index;

    // Insert a new slot if necessary.
    if (slotIndex >= slots.size()) {
      slots.resize(slotIndex + 1);
    }

    // get slot reference
    Slot &slot = slots.at(slotIndex);

    // Update the slot generation.
    slot.generation = key.generation;

    // Is the slot free?
    if (slot.free) {
      // set up slot
      // set slot to not free
      slot.free = false;
      slot.dataIndex = data.size();

      // push a new data member to the back of the data arrays.
      data.emplace_back();
      dataToSlotIndex.push_back(slotIndex);

      return data.back();

    } else {
      // Return the data.
      assert(dataToSlotIndex.at(slot.dataIndex) == slotIndex);

      return data.at(slot.dataIndex);
    }
  }

  /**
   * O(1)
   */
  void erase(const KeyType &key) {
    // check if there are enough slots
    if (slots.size() < key.index) {
      return;
    }

    const auto &slot = slots.at(key.index);

    if (slot.free) {
      return;
    }

    // Don't delete the data if the generations dont match.
    if (slot.generation != key.generation) {
      return;
    }

    assert(slot.dataIndex < data.size());
    assert(dataToSlotIndex.at(slot.dataIndex) == key.index);

    // swap the data to be deleted with the last data element.
    std::swap(data.at(slot.dataIndex), data.back());
    std::swap(dataToSlotIndex.at(slot.dataIndex), dataToSlotIndex.back());

    // fix the slot pointing to the swapped data
    slots.at(dataToSlotIndex.at(slot.dataIndex)).dataIndex = slot.dataIndex;

    // pop the data from the back
    data.pop_back();
    dataToSlotIndex.pop_back();

    // free the slot
    slots.at(key.index).free = true;
  }

  /**
   * O(1)
   * returns end() if key is invalid.
   */
  typename std::vector<DataType>::iterator at(const KeyType &key) {
    // check if there are enough slots
    if (slots.size() <= key.index) {
      return data.end();
    }

    const auto &slot = slots.at(key.index);

    // If the slot is earased return nothing.
    if (slot.free) {
      return data.end();
    }

    // Return nothing if the slot generation is different.
    if (slot.generation != key.generation) {
      return data.end();
    }

    assert(slot.dataIndex < data.size());

    return data.begin() + slot.dataIndex;
  }

  typename std::vector<DataType>::const_iterator at(const KeyType &key) const {
    return at(key);
  }

  /**
   * O(1)
   * iterator pointing to the beginning of this containers internal data
   * container.
   */
  typename std::vector<DataType>::iterator begin() { return data.begin(); }

  typename std::vector<DataType>::const_iterator begin() const {
    return data.begin();
  }

  /**
   * O(1)
   * iterator pointing to one past the end of this containers internal data
   * container.
   */
  typename std::vector<DataType>::iterator end() { return data.end(); }

  typename std::vector<DataType>::const_iterator end() const {
    return data.end();
  }

  /**
   * O(1)
   * number of elements currently stored the slot map.
   */
  size_t size() const { return data.size(); }

  /**
   * O(n)
   * Removes all data stored in this slot array.
   */
  void clear() {
    // Delete all data, but keep the memory.
    data.clear();
    dataToSlotIndex.clear();

    // Free all slots.
    for (auto &slot : slots) {
      slot.free = true;
    }
  }

  /**
   * O(1)
   * constructs a variable key index and generation from a data index.
   * Asserts that the data index is valid.
   */
  KeyType getKeyFromDataIndex(size_t dataIndex) const {
    assert(dataIndex < dataToSlotIndex.size());
    size_t slotIndex = dataToSlotIndex.at(dataIndex);

    KeyType result;

    result.index = slotIndex;
    assert(slotIndex < slots.size());
    result.generation = slots.at(slotIndex).generation;

    return result;
  }
};

}  // namespace ArgMin
