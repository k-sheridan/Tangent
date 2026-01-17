# SlotMap

The slot map is a data structure which provides constant time insert, erase, and access with a versioned key.

```cpp
class SlotMap<T, Key> {

SlotMap();

// O(1)
Key insert(T&& var);

// O(1)
void erase(Key key);

// O(1)
iterator at(Key key);

// O(1)
iterator begin();

// O(1)
iterator end();

// O(1) 
Key getKeyFromIterator(iterator);

};
```

## SlotMap Key Requirements

```cpp
struct KeyExample {
  unsigned index;
  unsigned generation;
}
```
