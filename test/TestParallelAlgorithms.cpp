
#include "argmin/ParallelAlgorithms/ParallelAlgorithms.h"
#include "argmin/Utilities/Logging.h"
#include <random>

#include <gtest/gtest.h>

TEST(ParallelAlgorithmsTest, parallel_transform)
{
    std::vector<long long> numbers(500);
    //std::generate(numbers.begin(), numbers.end(), std::rand);
    std::iota(numbers.begin(), numbers.end(), 1);
    //std::fill(numbers.begin(), numbers.end(), 10);

    std::vector<long long> result(numbers.size());

    const int wasteIter = 1000000;

    long long imoutside = 66777;

    auto fn = [imoutside](long long a) -> long long{for (int i = 0; i < wasteIter; ++i){a = std::max(a, (long long)i) + imoutside; } return a + 10; };

    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::SEQUENTIAL, numbers.begin(), numbers.end(),result.begin(), fn);
    std::vector<long long> result2(numbers.size());
    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::PARALLEL_CPU, numbers.begin(), numbers.end(),result2.begin(), fn);
    for (int i = 0; i < result.size(); ++i)
    {
        // LOG_INFO("Result: {}", result.at(i));
        EXPECT_EQ(result.at(i), result2.at(i));
    }
}

TEST(ParallelAlgorithmsTest, parallel_transform_binary)
{
    std::vector<long long> numbers(500);
     std::vector<long long> numbers2(500);
    //std::generate(numbers.begin(), numbers.end(), std::rand);
    std::iota(numbers.begin(), numbers.end(), 1);
    std::iota(numbers2.begin(), numbers2.end(), 1);
    //std::fill(numbers.begin(), numbers.end(), 10);

    std::vector<long long> result(numbers.size());

    const int wasteIter = 1000000;

    long long imoutside = 66777;

    auto fn = [imoutside](long long a, long long b) -> long long{for (int i = 0; i < wasteIter; ++i){a = std::max(a, (long long)i) + b + imoutside; } return a + b + 10; };

    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::SEQUENTIAL, numbers.begin(), numbers.end(), numbers2.begin(), result.begin(), fn);
    std::vector<long long> result2(numbers.size());
    QDVO::ParallelAlgorithms::transform(QDVO::ParallelAlgorithms::ExecutionType::PARALLEL_CPU, numbers.begin(), numbers.end(), numbers2.begin(), result2.begin(), fn);
    for (int i = 0; i < result.size(); ++i)
    {
        // LOG_INFO("Result: {}", result.at(i));
        EXPECT_EQ(result.at(i), result2.at(i));
    }
}

TEST(ParallelAlgorithmsTest, parallel_for_each)
{
    std::vector<long long> numbers(500);
    std::iota(numbers.begin(), numbers.end(), 1);
    std::vector<long long> moreNumbers(500);
    std::iota(moreNumbers.begin(), moreNumbers.end(), 1);

    const int wasteIter = 1000000;

    long long imoutside = 400;

    auto fn = [imoutside](long long& a) -> void {for (int i = 0; i < wasteIter; ++i){a = std::max(a, (long long)i) + imoutside; } };

    QDVO::ParallelAlgorithms::for_each(QDVO::ParallelAlgorithms::ExecutionType::SEQUENTIAL, numbers.begin(), numbers.end(), fn);
    QDVO::ParallelAlgorithms::for_each(QDVO::ParallelAlgorithms::ExecutionType::PARALLEL_CPU, moreNumbers.begin(), moreNumbers.end(), fn);
    for (int i = 0; i < numbers.size(); ++i)
    {
        // LOG_INFO("Result: {}", numbers.at(i));
        EXPECT_EQ(numbers.at(i), moreNumbers.at(i));
    }
}
