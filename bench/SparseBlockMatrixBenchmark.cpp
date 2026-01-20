#include <benchmark/benchmark.h>

#include "argmin/Types/SparseBlockRow.h"
#include "argmin/Types/SparseBlockMatrix.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/InverseDepth.h"

using namespace ArgMin;

template <typename ScalarType = double>
static void BM_SparseBlockRowDenseDotProduct(benchmark::State &state)
{
    SE3 pose;
    InverseDepth zinv;

    using SBR = ArgMin::SparseBlockRow<Scalar<ScalarType>, Dimension<2>, ArgMin::VariableGroup<SE3, InverseDepth>>;

    SBR sbr;

    VariableContainer<SE3, InverseDepth> variableContainer;

    // insert variables
    std::vector<VariableKey<SE3>> se3Keys;
    std::vector<VariableKey<InverseDepth>> dinvKeys;

    for (size_t i = 0; i < 100; ++i)
    {
        se3Keys.push_back(variableContainer.template getVariableMap<SE3>().insert(pose));
    }

    for (size_t i = 0; i < 400; ++i)
    {
        dinvKeys.push_back(variableContainer.template getVariableMap<InverseDepth>().insert(zinv));
    }

    //
    for (auto e : se3Keys)
    {
        Eigen::Matrix<ScalarType, 2, SE3::dimension> matrix;
        sbr.template getVariableMap<SE3>().insert(std::make_pair(e, matrix));
    }

    for (auto e : dinvKeys)
    {
        Eigen::Matrix<ScalarType, 2, InverseDepth::dimension> matrix;
        sbr.template getVariableMap<InverseDepth>().insert(std::make_pair(e, matrix));
    }

    Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> dx;

    dx.resize(1000, 1);

    Eigen::Matrix<ScalarType, 2, 1> result;

    for (auto _ : state)
    {
        sbr.dot(variableContainer, dx, result);
    }
}

template <typename ScalarType = double>
static void BM_SparseBlockRowDotProduct(benchmark::State &state)
{
    SE3 pose;
    InverseDepth zinv;

    using SBR = ArgMin::SparseBlockRow<Scalar<ScalarType>, Dimension<2>, ArgMin::VariableGroup<SE3, InverseDepth>>;

    SBR sbr;

    VariableContainer<SE3, InverseDepth> variableContainer;

    // insert variables
    std::vector<VariableKey<SE3>> se3Keys;
    std::vector<VariableKey<InverseDepth>> dinvKeys;

    for (size_t i = 0; i < 100; ++i)
    {
        se3Keys.push_back(variableContainer.template getVariableMap<SE3>().insert(pose));
    }

    for (size_t i = 0; i < 400; ++i)
    {
        dinvKeys.push_back(variableContainer.template getVariableMap<InverseDepth>().insert(zinv));
    }

    Eigen::Matrix<ScalarType, 2, SE3::dimension> matrix;
    sbr.template getVariableMap<SE3>().insert(std::make_pair(se3Keys.at(10), matrix));

    Eigen::Matrix<ScalarType, 2, InverseDepth::dimension> matrix2;
    sbr.template getVariableMap<InverseDepth>().insert(std::make_pair(dinvKeys.at(300), matrix2));

    Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> dx;

    dx.resize(1000, 1);

    Eigen::Matrix<ScalarType, 2, 1> result;

    for (auto _ : state)
    {
        sbr.dot(variableContainer, dx, result);
    }
}

BENCHMARK_TEMPLATE(BM_SparseBlockRowDenseDotProduct, double)->Unit(benchmark::kMicrosecond)->Iterations(100);
BENCHMARK_TEMPLATE(BM_SparseBlockRowDenseDotProduct, float)->Unit(benchmark::kMicrosecond)->Iterations(100);

BENCHMARK_TEMPLATE(BM_SparseBlockRowDotProduct, double)->Unit(benchmark::kNanosecond)->Iterations(100);
BENCHMARK_TEMPLATE(BM_SparseBlockRowDotProduct, float)->Unit(benchmark::kNanosecond)->Iterations(100);