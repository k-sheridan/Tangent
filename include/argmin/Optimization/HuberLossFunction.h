#pragma once

namespace ArgMin {
/**
 * The huber loss function is used for robust regression.
 * 
 * More information can be found here: https://en.wikipedia.org/wiki/Huber_loss
 * 
 * 
 * This weight is applied to the linearized error terms as follows:
 * 
 * \f$ \rho * J^{\top} \Sigma^{-1} J \f$
 * 
 * \f$ \rho * J^{\top} \Sigma^{-1} e \f$
 * 
 * Where \f$ \rho \f$ is the huber weight.
 */
template <typename ScalarType>
class HuberLossFunction
{
public:

    /**
     * Construct the loss function with the huber width = to c.
     */
    HuberLossFunction(ScalarType c_) {
        c = c_;
    } 

    ScalarType c;

    /**
     * Using the norm of the residual, compute the huber weight.
     */
    ScalarType computeWeight(ScalarType errorNorm, ScalarType errorNormSquared)
    {
        if (errorNorm < c) {
            return 1.0;
        } else {
            return c / errorNorm;
        }
    }
};

} // namespace ArgMin