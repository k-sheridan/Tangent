#pragma once

#include "argmin/Variables/OptimizableVariable.h"

namespace ArgMin
{

class SimpleScalar : public ArgMin::OptimizableVariable<double, 1>
{
public:

    double value;

    SimpleScalar() = default;

    SimpleScalar(double val) : value(val)
    {
    }

    void update(const Eigen::Matrix<double, 1, 1> &dx)
    {
        value += dx(0, 0);
    }
};

} // namespace ArgMin
