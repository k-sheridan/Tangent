
#include <gtest/gtest.h>

#include <numeric>

#include "argmin/Containers/SlotArray.h"
#include "argmin/Containers/SlotMap.h"

using namespace ArgMin;

template <typename T>
struct TypedKey : public SlotMapKeyBase {};

TEST(SlotArray, Simple) {
  using SA = SlotArray<int, TypedKey<int>>;

  SA map;

  TypedKey<int> key1, key2, key3;
  key1.index = 1;
  key2.index = 5;
  key3.index = 3;

  EXPECT_EQ(map.insert(key1, 10), SA::SUCCESS_NO_OVERWRITE);
  EXPECT_EQ(map.insert(key3, 30), SA::SUCCESS_NO_OVERWRITE);
  map.insert(key2) = 20;

  EXPECT_TRUE(*(map.at(key1)) == 10);
  EXPECT_TRUE(*(map.at(key2)) == 20);
  EXPECT_TRUE(*(map.at(key3)) == 30);

  EXPECT_TRUE(map.size() == 3);

  map.erase(key2);

  EXPECT_TRUE(*(map.at(key3)) == 30);
  EXPECT_TRUE(*(map.at(key1)) == 10);
  EXPECT_TRUE(map.size() == 2);

  EXPECT_EQ(map.at(key2), map.end());

  TypedKey<int> key4, key5;
  key4.index = 5;
  key5.index = 3;

  EXPECT_EQ(map.insert(key4, 20), SA::SUCCESS_NO_OVERWRITE);

  EXPECT_EQ(map.insert(key5, 50), SA::SUCCESS_OVERWRITE);

  EXPECT_TRUE(map.size() == 3);

  int i = 0;
  for (auto& e : map) {
    EXPECT_TRUE(e > 0);
    ++i;
  }
  EXPECT_EQ(i, 3);

  auto generatedKey = map.getKeyFromDataIndex(map.at(key5) - map.begin());

  EXPECT_EQ(generatedKey.index, key5.index);
}

TEST(SlotArray, VerifyGeneration) {
  using SA = SlotArray<int, TypedKey<int>>;
  using SM = SlotMap<double, TypedKey<int>>;

  SA array;
  SM map;

  auto key1 = map.insert(1.0);
  auto key2 = map.insert(2.0);

  array.insert(key1, 1);
  array.insert(key2, 2);

  map.erase(key2);
  auto key3 = map.insert(3);
  ASSERT_EQ(key2.index, key3.index);
  EXPECT_NE(key2.generation, key3.generation);

  array.erase(key2);
  array.insert(key3, 3);

  auto generatedKey = array.getKeyFromDataIndex(array.at(key3) - array.begin());

  // Now the generated key should work with the slot map.
  EXPECT_NE(map.at(generatedKey), map.end());
}
