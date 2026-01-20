
#include "argmin/Containers/SlotMap.h"
#include <numeric>

#include <gtest/gtest.h>

using namespace ArgMin;

template <typename T>
struct TypedKey : public SlotMapKeyBase {
    using SlotMapKeyBase::index_type;
    using SlotMapKeyBase::generation_type;
};

TEST(SlotMap, Simple) {

    SlotMap<int, TypedKey<int>> map;

    auto key1 = map.insert(10);
    auto key2 = map.insert(20);
    auto key3 = map.insert(30);

    EXPECT_TRUE(*(map.at(key1)) == 10);
    EXPECT_TRUE(*(map.at(key2)) == 20);
    EXPECT_TRUE(*(map.at(key3)) == 30);

    EXPECT_TRUE(map.size() == 3);

    map.erase(key2);

    EXPECT_TRUE(*(map.at(key3)) == 30);
    EXPECT_TRUE(*(map.at(key1)) == 10);
    EXPECT_TRUE(map.size() == 2);

    EXPECT_EQ(map.at(key2), map.end());

    auto key4 = map.insert(20);

    EXPECT_EQ(key4.generation, 1);
    EXPECT_EQ(key4.index, 1);

    auto key5 = map.insert(40);

    EXPECT_TRUE(map.size() == 4);
    EXPECT_EQ(key5.generation, 0);
    EXPECT_EQ(key5.index, 3);

    int i = 0;
    for (auto& e : map)
    {
        EXPECT_TRUE(e > 0);
        ++i;
    }
    EXPECT_EQ(i, 4);

    auto generatedKey = map.getKeyFromDataIndex(map.at(key5) - map.begin());

    EXPECT_EQ(generatedKey.index, key5.index);
    EXPECT_EQ(generatedKey.generation, key5.generation);
}

TEST(SlotMap, Stress) {
    std::vector<int> values(1000000);
    std::iota(values.begin(), values.end(), 0);

    SlotMap<int, TypedKey<int>> map;

    std::vector<TypedKey<int>> keys;

    for (size_t idx = 0; idx < values.size(); ++idx)
    {
        keys.push_back(map.insert(values.at(idx)));
    }

    for (size_t idx = 0; idx < keys.size(); ++idx)
    {
        EXPECT_EQ(*(map.at(keys.at(idx))), values.at(idx));
    }

    values.erase(values.begin() + 100);
    map.erase(keys.at(100));
    keys.erase(keys.begin() + 100);

    for (size_t idx = 0; idx < keys.size(); ++idx)
    {
        EXPECT_EQ(*(map.at(keys.at(idx))), values.at(idx));
    }
}

