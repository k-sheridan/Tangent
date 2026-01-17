
#include "argmin/ParallelAlgorithms/ParallelAlgorithms.h"
#include <random>

#include <gtest/gtest.h>

TEST(ParallelAlgorithmsTest, parallel_transform)
{
    std::vector<int> numbers(500);
    //std::generate(numbers.begin(), numbers.end(), std::rand);
    std::iota(numbers.begin(), numbers.end(), 1);
    //std::fill(numbers.begin(), numbers.end(), 10);

    std::vector<int> result(numbers.size());

    const int wasteIter = 1000000;

    int imoutside = 66777;

    auto fn = [imoutside](int a) -> int{for (int i = 0; i < wasteIter; ++i){a = std::max(a, i) + imoutside; } return a + 10; };

    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::SEQUENTIAL, numbers.begin(), numbers.end(),result.begin(), fn);
    std::vector<int> result2(numbers.size());
    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::PARALLEL_CPU, numbers.begin(), numbers.end(),result2.begin(), fn);
    for (int i = 0; i < result.size(); ++i)
    {
        //std::cout << result.at(i) << std::endl;
        EXPECT_EQ(result.at(i), result2.at(i));
    }
}

TEST(ParallelAlgorithmsTest, parallel_transform_binary)
{
    std::vector<int> numbers(500);
     std::vector<int> numbers2(500);
    //std::generate(numbers.begin(), numbers.end(), std::rand);
    std::iota(numbers.begin(), numbers.end(), 1);
    std::iota(numbers2.begin(), numbers2.end(), 1);
    //std::fill(numbers.begin(), numbers.end(), 10);

    std::vector<int> result(numbers.size());

    const int wasteIter = 1000000;

    int imoutside = 66777;

    auto fn = [imoutside](int a, int b) -> int{for (int i = 0; i < wasteIter; ++i){a = std::max(a, i) + b + imoutside; } return a + b + 10; };

    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::SEQUENTIAL, numbers.begin(), numbers.end(), numbers2.begin(), result.begin(), fn);
    std::vector<int> result2(numbers.size());
    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::PARALLEL_CPU, numbers.begin(), numbers.end(), numbers2.begin(), result2.begin(), fn);
    for (int i = 0; i < result.size(); ++i)
    {
        //std::cout << result.at(i) << std::endl;
        EXPECT_EQ(result.at(i), result2.at(i));
    }
}

TEST(ParallelAlgorithmsTest, parallel_for_each)
{
    std::vector<int> numbers(500);
    std::iota(numbers.begin(), numbers.end(), 1);
    std::vector<int> moreNumbers(500);
    std::iota(moreNumbers.begin(), moreNumbers.end(), 1);

    const int wasteIter = 1000000;

    int imoutside = 400;

    auto fn = [imoutside](int& a) -> void {for (int i = 0; i < wasteIter; ++i){a = std::max(a, i) + imoutside; } };

    QDVO::ParallelAlgorithms::for_each(QDVO::ParallelAlgorithms::ExecutionType::SEQUENTIAL, numbers.begin(), numbers.end(), fn);
    QDVO::ParallelAlgorithms::for_each(QDVO::ParallelAlgorithms::ExecutionType::PARALLEL_CPU, moreNumbers.begin(), moreNumbers.end(), fn);
    for (int i = 0; i < numbers.size(); ++i)
    {
        //std::cout << result.at(i) << std::endl;
        EXPECT_EQ(numbers.at(i), moreNumbers.at(i));
    }
}
