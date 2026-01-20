#pragma once

#include <vector>
#include <set>
#include <cassert>
#include <cstddef>
#include <limits>

namespace ArgMin
{

struct SlotMapKeyBase
{
public:
    using index_type = size_t;
    using generation_type = size_t;

    index_type index = std::numeric_limits<index_type>::max();
    generation_type generation = 0;

    void setInvalid()
    {
        index = std::numeric_limits<index_type>::max();
    }

    bool isInvalid()
    {
        return index == std::numeric_limits<index_type>::max();
    }
};

template <typename T>
struct TypedSlotMapKey : public SlotMapKeyBase {
    /// A compile time helper to get the variable type of this key.
    typedef T variable_type;

    /// Compares two keys by their index and generation.
    bool operator==(const TypedSlotMapKey<T> &other) const
    {
        return this->index == other.index && this->generation == other.generation;
    }
};

/**
 * This is a slot map as defined by allan deutsch.
 */
template <typename DataType, typename KeyType = TypedSlotMapKey<DataType>>
class SlotMap
{

    static_assert(std::is_integral<typename KeyType::index_type>::value);
    static_assert(std::is_integral<typename KeyType::generation_type>::value);

    struct Slot
    {
        size_t dataIndex; // index of the data in the data array.
        typename KeyType::generation_type generation = 0;
        bool free = true;
    };

    std::vector<Slot> slots;       // Set of keys currently in use.
    std::vector<size_t> freeSlots; // Stores indices of which slots can be inserted to.

    std::vector<DataType> data;          // vector containing all data.
    std::vector<size_t> dataToSlotIndex; // vector which points data to its slot index.

public:
    typedef KeyType key_type;
    typedef DataType data_type;

    SlotMap()
    {
    }

    /// Emptys the slot map while retaining its memory.
    void clear() {
        slots.clear();
        freeSlots.clear();
        data.clear();
        dataToSlotIndex.clear();
    }

    /**
     * O(1)
     */
    KeyType insert(DataType value)
    {
        size_t slotIndex;

        if (freeSlots.empty())
        {
            // add new slot
            slotIndex = slots.size();
            slots.emplace_back();
        }
        else
        {
            slotIndex = freeSlots.back();
            freeSlots.pop_back();
        }

        // get slot reference
        Slot &slot = slots.at(slotIndex);
        assert(slot.free == true);

        // set up slot
        // set slot to not free
        slot.free = false;
        slot.dataIndex = data.size();

        // push a new data member to the back of the data arrays.
        data.push_back(std::move(value));
        dataToSlotIndex.push_back(slotIndex);

        // setup key.
        KeyType key;
        key.index = slotIndex;
        key.generation = slot.generation;

        return key;
    }

    /**
     * O(1)
     */
    void erase(const KeyType &key)
    {
        // check if there are enough slots
        if (slots.size() <= key.index)
        {
            return;
        }

        const auto &slot = slots.at(key.index);

        // check if the generations match
        if (slot.generation != key.generation)
        {
            return;
        }

        if (slot.free)
        {
            return;
        }

        assert(slot.dataIndex < data.size());

        // swap the data to be deleted with the last data element.
        std::swap(data.at(slot.dataIndex), data.back());
        std::swap(dataToSlotIndex.at(slot.dataIndex), dataToSlotIndex.back());

        // fix the slot pointing to the swapped data
        slots.at(dataToSlotIndex.at(slot.dataIndex)).dataIndex = slot.dataIndex;

        // increment the deleted slots generation
        const size_t deletedSlotIndex = dataToSlotIndex.back();
        ++slots.at(deletedSlotIndex).generation;

        // pop the data from the back
        data.pop_back();
        dataToSlotIndex.pop_back();

        // free the slot
        slots.at(deletedSlotIndex).free = true;
        freeSlots.push_back(deletedSlotIndex);
    }

    /**
     * O(1)
     * returns end() if key is invalid.
     */
    typename std::vector<DataType>::iterator at(const KeyType &key)
    {
        // check if there are enough slots
        if (slots.size() <= key.index)
        {
            return data.end();
        }

        const auto &slot = slots.at(key.index);

        // check if the generations match
        if (slot.generation != key.generation)
        {
            return data.end();
        }

        if (slot.free)
        {
            return data.end();
        }

        assert(slot.dataIndex < data.size());

        return data.begin() + slot.dataIndex;
    }

    typename std::vector<DataType>::const_iterator at(const KeyType &key) const
    {
        return at(key);
    }

    /**
     * O(1)
     * iterator pointing to the beginning of this containers internal data container.
     */
    typename std::vector<DataType>::iterator begin()
    {
        return data.begin();
    }

    typename std::vector<DataType>::const_iterator begin() const
    {
        return data.begin();
    }

    /**
     * O(1)
     * iterator pointing to one past the end of this containers internal data container.
     */
    typename std::vector<DataType>::iterator end()
    {
        return data.end();
    }

    typename std::vector<DataType>::const_iterator end() const
    {
        return data.end();
    }

    /**
     * O(1)
     * number of elements currently stored the slot map.
     */
    size_t size() const
    {
        return data.size();
    }

    /**
     * O(1)
     * 
     * Increments the generation of a slot to invalidate all previously returned keys.
     * The key must be a valid key, otherwise an invalid key is returned.
     * 
     * Equivalent to:
     * data = at(key1);
     * erase(key1);
     * key2 = insert(data);
     */
    KeyType updateSlotGeneration(const KeyType& key)
    {
        // check if there are enough slots
        if (slots.size() <= key.index)
        {
            return KeyType();
        }

        auto &slot = slots.at(key.index);

        // check if the generations match
        if (slot.generation != key.generation)
        {
            return KeyType();
        }

        if (slot.free)
        {
            return KeyType();
        }

        KeyType newKey = key;

        ++slot.generation;
        newKey.generation = slot.generation;

        return newKey;
    }

    /**
     * O(1)
     * constructs a variable key index and generation from a data index.
     * Asserts that the data index is valid.
     */
    KeyType getKeyFromDataIndex(size_t dataIndex)
    {
        assert(dataIndex < dataToSlotIndex.size());
        size_t slotIndex = dataToSlotIndex.at(dataIndex);

        KeyType result;

        result.index = slotIndex;
        assert(slotIndex < slots.size());
        result.generation = slots.at(slotIndex).generation;

        return result;
    }
};

} //namespace ArgMin