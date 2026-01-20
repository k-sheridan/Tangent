#pragma once

#include <cassert>
#include <cmath>
#include <numeric>
#include <type_traits>

#include "argmin/Types/BlockVector.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Types/GaussianPrior.h"
#include "argmin/Optimization/HuberLossFunction.h"
#include "argmin/Utilities/Logging.h"
#include "argmin/Types/MetaHelpers.h"
#include "argmin/ParallelAlgorithms/ParallelAlgorithms.h"
#include "argmin/Utilities/Profiling.h"
#include "argmin/Containers/SlotArray.h"
#include "argmin/Containers/SlotMap.h"

namespace ArgMin {

template <typename...>
class PSDSchurSolver;

/**
 * This class provides the functions to solve a levenberg marquardt or gauss
 * newton iteration. It sets up the sparse linear system, and solves it by
 * exploiting the sparsity using the schur complement.
 *
 * Internally, the linear system, which contains a square PSD matrix, is split
 * into 3 blocks
 *
 * \f$ \left[
 * \begin{array}{c|c}
 * A & B \\
 * \hline
 * B^{T} & D
 * \end{array}
 * \right] \f$
 *
 * A is a dense matrix, D is a block diagonal matrix, and B is a dense matrix
 * correlating A and D.
 *
 * Uncorrelated variables are variables which share no off diagonal elements.
 * The uncorrelated variable set must be a subset of all variables.
 */
template <typename ScalarType, typename LossFunctionType,
          template <typename...> class ErrorTermGroup,
          template <typename...> class VariableGroup, typename... ErrorTerms,
          typename... Variables, typename... UncorrelatedVariables>
class PSDSchurSolver<Scalar<ScalarType>, LossFunction<LossFunctionType>,
                     ErrorTermGroup<ErrorTerms...>, VariableGroup<Variables...>,
                     VariableGroup<UncorrelatedVariables...>> {
 public:
  using RHSBlockVector = BlockVector<Scalar<ScalarType>, Dimension<1>,
                                     VariableGroup<UncorrelatedVariables...>>;
  using DxBlockVector = BlockVector<Scalar<ScalarType>, Dimension<1>,
                                    VariableGroup<Variables...>>;
  template <typename VariableType>
  using BVector = SlotArray<
      Eigen::Matrix<ScalarType, Eigen::Dynamic, VariableType::dimension>,
      VariableKey<VariableType>>;
  template <typename VariableType>
  using DBlock = Eigen::Matrix<ScalarType, VariableType::dimension,
                               VariableType::dimension>;
  template <typename VariableType>
  using DVector = SlotArray<DBlock<VariableType>, VariableKey<VariableType>>;
  template <typename VariableType>
  using IndexMap = SlotArray<size_t, VariableKey<VariableType>>;
  using LossFunction = HuberLossFunction<ScalarType>;

  struct Settings {
    /// The first lambda used during the solve.
    double initialLambda = 1e6;
    /// The value lambda is divided by each time a successful iteration occurs.
    double lambdaReductionMultiplier = 10;
    /// The maximum number of iterations for the solve.
    int maximumIterations = 25;
    /// Stop after the error increases.
    bool stopAfterErrorIncrease = false;
    /// Allow the solver to parallelize linearization.
    bool parallelizeErrorTermLinearization = false;
    /// elements per thread cost model.
    /// This is used to determine how many threads to use for parallelization of
    /// error terms. This is meant to reduce the overhead of spawning n threads.
    double errorTermsPerThread = 1000;
    /// Stopping condition based on the length of the update.
    double updateThreshold = 0;
    /// Stopping condition based on the change in error.
    double errorDeltaThreshold = 0;
  } settings;

  struct SolveResult {
    /// The error at each iteration.
    std::vector<ScalarType> whitenedSqError;
  };

  /// Loss function used to weight the error for each error term.
  LossFunctionType lossFunction;

  /// Dense matrix in the upper right corner
  Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> A;
  /// Top right block matrix. Equal to bottom left transposed.
  std::tuple<BVector<UncorrelatedVariables>...> B;
  /// Used during the schur solve to store a precomputed: \f$ -B D^{-1} \f$
  std::tuple<BVector<UncorrelatedVariables>...> negativeBDinv;
  /// Block diagonal matrix in the bottom right. Each block is invertible.
  std::tuple<DVector<UncorrelatedVariables>...> D;
  /// Pre allocated solution vector.
  Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> dx;
  DxBlockVector dxBlockVector;
  /// Preallocated rhs vector.
  Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> b_correlated;
  /// preallocated uncorreleted part of the b vector.
  RHSBlockVector b_uncorrelated;
  /// Tuple of slot arrays which store the current index of a given variable in
  /// A.
  std::tuple<IndexMap<Variables>...> variableToIndexMaps;
  /// The current dimension of the A matrix. This exists because we do not need
  /// to reduce the size of A.
  size_t dimensionOfA = 0;
  /// The current total problem dimension.
  size_t totalDimension = 0;

  PSDSchurSolver(LossFunctionType &lossFunction) : lossFunction(lossFunction) {}

  /**
   * This function is ran once before a solve. This could be used to preallocate
   * memory or precompute values.
   */
  void initialize(VariableContainer<Variables...> &variables,
                  ErrorTermContainer<ErrorTerms...> &errorTerms) {
    LOG_TRACE("Initializing Solver.");

    PROFILE(std::string(typeid(errorTerms).name()) + "_initializeSolver");

    removeOldVariablesFromSlotArrays(variables);

    addNewVariablesToSlotArrays(variables);

    precomputeIndexMapAndResizeMatrices(variables);

    updateVariablePointers(variables, errorTerms);
  }

  /**
   * This function will linearize the error terms if necessary, build the
   * problem, and solve for the perturbation while exploiting the knowledge of
   * the uncorrelated variable set.
   *
   * After the solve, it can be assumed that the error terms are in the
   * linearized state used for the final iteration.
   */
  template <typename GaussianPriorType>
  SolveResult solveLevenbergMarquardt(
      VariableContainer<Variables...> &variables,
      ErrorTermContainer<ErrorTerms...> &errorTerms, GaussianPriorType &prior) {
    LOG_TRACE("Starting Levenberg-Marquardt solve.");
    PROFILE(std::string(typeid(errorTerms).name()) +
            "_solveLevenbergMarquardt");
    // Initialize the solver.
    initialize(variables, errorTerms);

    SolveResult result;
    ScalarType lambda = settings.initialLambda;

    // Set up the solver for the first iteration.
    LOG_TRACE("Linearizing error terms for first iteration.");
    // Linearize the error terms.
    linearize(variables, errorTerms);

    // Iterate and solve.
    for (int iteration = 0; iteration < settings.maximumIterations;
         ++iteration) {
      LOG_TRACE("Building linear system.");
      // Build the linear system.
      double whitenedSqErrorBeforeSolve =
          buildLinearSystem(prior, errorTerms, variables);

      LOG_TRACE("Adding lambda to linear system.");
      // Add lambda to the linear system.
      addLambdaToLinearSystem(lambda);

      LOG_TRACE("Iteration: {}  Whitened Squared Error: {} Lambda: {}",
                iteration, whitenedSqErrorBeforeSolve, lambda);

      // Add the current error to the error array.
      result.whitenedSqError.push_back(whitenedSqErrorBeforeSolve);

      LOG_TRACE("Solving");
      // Solve the linear system.
      solveLinearSystem(variables, errorTerms, prior);

      // Verify that the perturbation is not nan.
      if (isUpdateValid()) {
        // Make sure that after applying this update, it can be reverted.
        ensureUpdateIsRevertible(variables);

        // Apply the update.
        LOG_TRACE("Updating variables to check if error increased.");
        applyUpdateToVariables(variables);

        // Linearize the error terms.
        LOG_TRACE("Linearizing error terms.");
        linearize(variables, errorTerms);
        // Compute the error with this update.
        double errorAfterUpdate =
            computeWhitenedSqError(errorTerms, variables, false);
        LOG_TRACE("Whitened Squared Error after update: {}", errorAfterUpdate);

        if (errorAfterUpdate < whitenedSqErrorBeforeSolve) {
          // The error decreased, reduce lambda.
          LOG_TRACE("Error decreased... reducing lambda.");
          lambda = lambda / settings.lambdaReductionMultiplier;

          LOG_TRACE("Updating prior after successful update.");
          prior.update(dxBlockVector);
        } else {
          // The error increased or stagnated, break.
          LOG_TRACE(
              "Error increased... increasing lambda and reverting update.");
          lambda = lambda * settings.lambdaReductionMultiplier;

          // Revert the previous update.
          applyUpdateToVariables<true>(variables);

          // Stop if requested.
          if (settings.stopAfterErrorIncrease) {
            LOG_TRACE("Stopping.");
            break;
          } else {
            // Linearize the error terms.
            LOG_TRACE("Linearizing error terms.");
            linearize(variables, errorTerms);
          }
        }

        // Evaluate stopping conditons.
        //
        // Check if the change in error is too small.
        double dErrorSquared =
            std::abs(errorAfterUpdate - whitenedSqErrorBeforeSolve);
        LOG_TRACE("Change in squared error {}", dErrorSquared);
        if (dErrorSquared < settings.errorDeltaThreshold) {
          LOG_TRACE(
              "Stopping condition reached. Change in squared error too small.");
          break;
        }

        // Check if the update was too small.
        double updateLength = computeUpdateLength();
        LOG_TRACE("norm of dx = {}", updateLength);
        if (updateLength < settings.updateThreshold) {
          LOG_TRACE("Update too small stopping optimization.");
          break;
        }

      } else {
        // Return early without updating
        LOG_ERROR("Perturbation invalid, returning early");
        return result;
      }
    }

    LOG_TRACE("Finished Levenberg-Marquardt solve.");
    return result;
  }

  /**
   * This adds a lambda to the diagonal members of A.
   */
  void addLambdaToLinearSystem(ScalarType lambda) {
    A.diagonal().array() += lambda;

    internal::static_for(D, [&](auto i, auto &blockArray) {
      for (auto &block : blockArray) {
        block.diagonal().array() += lambda;
      }
    });
  }

  /**
   * Computes the norm of the update vector.
   */
  double computeUpdateLength() {
    double squaredNorm = 0;
    std::tuple<Variables *...> tupleOfVars;
    internal::static_for(tupleOfVars, [&](auto i, auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;
      for (const auto &dx : dxBlockVector.template getRowMap<ThisVariable>()) {
        squaredNorm += dx.squaredNorm();
      }
    });
    return sqrt(squaredNorm);
  }

  /**
   * Verifies that the update vector, dx, is valid.
   */
  bool isUpdateValid() {
    bool result = true;
    std::tuple<Variables *...> tupleOfVars;
    internal::static_for(tupleOfVars, [&](auto i, auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;
      for (const auto &dx : dxBlockVector.template getRowMap<ThisVariable>()) {
        bool localResult = true;
        for (int i = 0; i < dx.rows(); ++i) {
          if (std::isnan(dx(i, 0))) {
            localResult = false;
          }
        }
        if (localResult == false) {
          LOG_ERROR("Update was invalid for a {} with an update of \n{}\n",
                    typeid(ThisVariable).name(), dx);
        }
        result = localResult && result;
      }
    });
    return result;
  }

  /**
   * Evaluates the error terms with the current variables.
   */
  double computeWhitenedSqError(ErrorTermContainer<ErrorTerms...> &errorTerms,
                                VariableContainer<Variables...> &variables,
                                bool relinearize = true) {
    PROFILE(std::string(typeid(ErrorTermContainer<ErrorTerms...>).name()) +
            "_computeWhitenedSquaredError");

    int nErrorTerms = 0;
    double whitenedSqError = 0;
    // iterate through all error terms
    internal::static_for(
        errorTerms.tupleOfErrorTermMaps,
        [&](auto errorTermTypeIndex, auto &errorTermMap) {
          LOG_TRACE(
              "Evaluating error with Error Term Type: {}",
              typeid(
                  typename std::tuple_element<errorTermTypeIndex,
                                              std::tuple<ErrorTerms...>>::type)
                  .name());
          for (auto &errorTerm : errorTermMap) {
            // TODO Give evaluate the ability to mark if a residual is invalid.
            if (relinearize) {
              errorTerm.evaluate(variables, true);
            }

            // Check if the linearization is valid for this error term.
            if (errorTerm.linearizationValid) {
              double sqError = errorTerm.residual.squaredNorm();
              double error = sqrt(sqError);
              double weight = lossFunction.computeWeight(error, sqError);

              whitenedSqError += sqError;
              ++nErrorTerms;
            }
          }
        });
    LOG_TRACE("Computed whitened squared error with {} valid error terms.",
              nErrorTerms);
    return whitenedSqError / nErrorTerms;
  }

  /**
   * Ensures that the update vector can be reverted if an update was bad.
   * @param Variables which will be updated.
   */
  void ensureUpdateIsRevertible(VariableContainer<Variables...> &variables) {
    internal::static_for(variables.tupleOfVariableMaps, [&](auto i,
                                                            auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      for (auto it = variableMap.begin(); it != variableMap.end(); it++) {
        auto &variable = *(it);

        auto key = variableMap.getKeyFromDataIndex(it - variableMap.begin());

        assert(dxBlockVector.blockExists(key));

        variable.ensureUpdateIsRevertible(dxBlockVector.getRowBlock(key));
      }
    });
  }

  /**
   * Applies the perturbation to all variables using their box plus operator.
   * Assumes that the linear system has just been solved.
   *
   * @tparam revert If true, this will apply the reverse update to the
   * variables. This can be used to revert a negative update.
   */
  template <bool Revert = false>
  void applyUpdateToVariables(VariableContainer<Variables...> &variables) {
    PROFILE(std::string(typeid(ErrorTermContainer<ErrorTerms...>).name()) +
            "_applyUpdate");

    internal::static_for(variables.tupleOfVariableMaps, [&](auto i,
                                                            auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      for (auto it = variableMap.begin(); it != variableMap.end(); it++) {
        auto &variable = *(it);

        auto key = variableMap.getKeyFromDataIndex(it - variableMap.begin());

        assert(dxBlockVector.blockExists(key));

        if constexpr (Revert) {
          variable.update(-dxBlockVector.getRowBlock(key));
        } else {
          variable.update(dxBlockVector.getRowBlock(key));
        }
      }
    });
  }

  /**
   * Solves the linear system currently setup.
   * Warning the linear system is invalidated after this runs.
   * It is assumed that the error terms are linearized.
   *
   * From: https://en.wikipedia.org/wiki/Schur_complement
   *
   * \f$ {\displaystyle M=\left[{\begin{matrix}A&B\\C&D\end{matrix}}\right]} \f$
   *
   * \f$ {\displaystyle M/D:=A-BD^{-1}C\,} \f$
   *
   * \f$ {\displaystyle M/A:=D-CA^{-1}B.} \f$
   *
   * \f$ {\displaystyle
   * {\begin{aligned}&{\begin{bmatrix}A&B\\C&D\end{bmatrix}}^{-1}={\begin{bmatrix}I_{p}&0\\-D^{-1}C&I_{q}\end{bmatrix}}{\begin{bmatrix}\left(A-BD^{-1}C\right)^{-1}&0\\0&D^{-1}\end{bmatrix}}{\begin{bmatrix}I_{p}&-BD^{-1}\\0&I_{q}\end{bmatrix}}\\[4pt]={}&{\begin{bmatrix}\left(A-BD^{-1}C\right)^{-1}&-\left(A-BD^{-1}C\right)^{-1}BD^{-1}\\-D^{-1}C\left(A-BD^{-1}C\right)^{-1}&D^{-1}+D^{-1}C\left(A-BD^{-1}C\right)^{-1}BD^{-1}\end{bmatrix}}\\[4pt]={}&{\begin{bmatrix}\left(A-BD^{-1}C\right)^{-1}&-\left(A-BD^{-1}C\right)^{-1}BD^{-1}\\-D^{-1}C\left(A-BD^{-1}C\right)^{-1}&\left(D-CA^{-1}B\right)^{-1}\end{bmatrix}}\\[4pt]={}&{\begin{bmatrix}\left(M/D\right)^{-1}&-\left(M/D\right)^{-1}BD^{-1}\\-D^{-1}C\left(M/D\right)^{-1}&\left(M/A\right)^{-1}\end{bmatrix}}.\end{aligned}}}
   * \f$
   *
   */
  template <typename GaussianPriorType>
  void solveLinearSystem(
      VariableContainer<Variables...> &variables,
      ErrorTermContainer<ErrorTerms...> &linearizedErrorTerms,
      GaussianPriorType &prior) {
    PROFILE(std::string(typeid(linearizedErrorTerms).name()) +
            "_solveLinearSystem");

    LOG_TRACE(
        "Starting Schur Solve with problem dimension: {} and a correlated "
        "dimension of: {}",
        totalDimension, dimensionOfA);
    LOG_TRACE("Computing D^{-1}");
    // Solve for the deltas using the Schur Complement.
    // First invert the D matrix
    internal::static_for(D, [&](auto i, auto &matrixSlotArray) {
      // Get the variable for this section of the matrix.
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type RowVariable;

      for (Eigen::Matrix<ScalarType, RowVariable::dimension,
                         RowVariable::dimension> &matrix : matrixSlotArray) {
        // TODO Maybe avoid the copy.
        matrix = matrix.inverse();
      }
    });

    LOG_TRACE("Computing -B D^{-1}");
    // Compute -B Dinv, and compute the inverse Schur Complement of D.
    // This will use A to store the schur complement before inversion.
    internal::static_for(B, [&](auto i, auto &matrixSlotArray) {
      // Get the variable for this section of the matrix.
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type RowVariable;
      for (auto it = matrixSlotArray.begin(); it != matrixSlotArray.end();
           it++) {
        // The original b matrix.
        const Eigen::Matrix<ScalarType, Eigen::Dynamic, RowVariable::dimension>
            &bMatrix = *(it);

        auto key =
            matrixSlotArray.getKeyFromDataIndex(it - matrixSlotArray.begin());
        assert(variables.variableExists(key));

        // At this point Dinv should have been computed.
        auto dinvIt = std::get<DVector<RowVariable>>(D).at(key);
        assert(dinvIt != std::get<DVector<RowVariable>>(D).end());
        const Eigen::Matrix<ScalarType, RowVariable::dimension,
                            RowVariable::dimension> &dinv = *(dinvIt);

        // Get the matrix we are going to compute.
        auto negativeBDinvMatrixIt =
            std::get<BVector<RowVariable>>(negativeBDinv).at(key);
        assert(negativeBDinvMatrixIt !=
               std::get<BVector<RowVariable>>(negativeBDinv).end());
        Eigen::Matrix<ScalarType, Eigen::Dynamic, RowVariable::dimension>
            &negativeBDinvMatrix = *(negativeBDinvMatrixIt);

        // It is possible that this matrix has more rows than needed.
        negativeBDinvMatrix.noalias() = (bMatrix * -dinv).eval();

        // Add BDinvB' to A.
        A.noalias() += (negativeBDinvMatrix * bMatrix.transpose()).eval();
      }
    });

    // At this point we have computed the inverse of the LHS.
    // Now we just have to multiply our results with the RHS.

    LOG_TRACE("Computing -B D^{-1} b_{uncorrelated}");
    // Multiply -BDinv * b_uncorrelated.
    internal::static_for(negativeBDinv, [&](auto i, auto &matrixSlotArray) {
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type RowVariable;
      for (auto it = matrixSlotArray.begin(); it != matrixSlotArray.end();
           it++) {
        const Eigen::Matrix<ScalarType, Eigen::Dynamic, RowVariable::dimension>
            &negativeBDinvMatrix = *(it);

        auto key =
            matrixSlotArray.getKeyFromDataIndex(it - matrixSlotArray.begin());
        assert(variables.variableExists(key));

        const Eigen::Matrix<ScalarType, RowVariable::dimension, 1>
            &rhsBlockMatrix = b_uncorrelated.getRowBlock(key);

        b_correlated.noalias() += (negativeBDinvMatrix * rhsBlockMatrix).eval();
      }
    });

    LOG_TRACE(
        "Computing dx_{correlated} = (A - B D^{-1} B^{T})^{-1} b_{correlated}");
    // Multiply the inverse schur complement of D by the correlated b vector.
    dx.block(0, 0, dimensionOfA, 1) = A.ldlt().solve(b_correlated);

    LOG_TRACE("Computing D^{-1} b_{uncorrelated}");
    // Multiply Dinv by the b_uncorrelated vector.
    internal::static_for(D, [&](auto i, auto &matrixSlotArray) {
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type RowVariable;
      for (auto it = matrixSlotArray.begin(); it != matrixSlotArray.end();
           it++) {
        // D should be inverted at this point.
        const Eigen::Matrix<ScalarType, RowVariable::dimension,
                            RowVariable::dimension> &DinvMatrix = *(it);

        auto key =
            matrixSlotArray.getKeyFromDataIndex(it - matrixSlotArray.begin());
        assert(variables.variableExists(key));

        const Eigen::Matrix<ScalarType, Eigen::Dynamic, RowVariable::dimension>
            &bMatrixBlock = b_uncorrelated.getRowBlock(key);

        auto &indexMap = std::get<IndexMap<RowVariable>>(variableToIndexMaps);
        auto indexIt = indexMap.at(key);
        assert(indexIt != indexMap.end());

        dx.template block<RowVariable::dimension, 1>(*(indexIt), 0).noalias() =
            (DinvMatrix * bMatrixBlock).eval();
      }
    });

    LOG_TRACE("Computing dx_{uncorrelated} = -B D^{-1} dx_{correlated}");
    // At this point the partial solution is stored in the dx vector.
    // Compute the final sweep of (-BDinv)^T * dx_uncorrelated.
    // This is correct  because Dinv is symmetric, and C = B^T
    internal::static_for(negativeBDinv, [&](auto i, auto &matrixSlotArray) {
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type RowVariable;
      for (auto it = matrixSlotArray.begin(); it != matrixSlotArray.end();
           it++) {
        const Eigen::Matrix<ScalarType, Eigen::Dynamic, RowVariable::dimension>
            &negativeBDinvMatrix = *(it);

        auto key =
            matrixSlotArray.getKeyFromDataIndex(it - matrixSlotArray.begin());
        assert(variables.variableExists(key));

        auto &indexMap = std::get<IndexMap<RowVariable>>(variableToIndexMaps);
        auto indexIt = indexMap.at(key);
        assert(indexIt != indexMap.end());

        dx.template block<RowVariable::dimension, 1>(*(indexIt), 0).noalias() +=
            (negativeBDinvMatrix.transpose() * dx.block(0, 0, dimensionOfA, 1))
                .eval();
      }
    });

    LOG_TRACE("Setting the dx block vector");
    // Set the dx block vector from the index map and dx vector
    internal::static_for(variableToIndexMaps, [&](auto i, auto &indexMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      for (auto it = indexMap.begin(); it != indexMap.end(); it++) {
        auto key = indexMap.getKeyFromDataIndex(it - indexMap.begin());

        assert(dxBlockVector.blockExists(key));

        Eigen::Matrix<ScalarType, ThisVariable::dimension, 1> &block =
            dxBlockVector.getRowBlock(key);

        block = dx.template block<ThisVariable::dimension, 1>(*(it), 0);
      }
    });

    LOG_TRACE("Schur Solve complete. ");
  }

  /// Linearizes all error terms stored in this container.
  void linearize(VariableContainer<Variables...> &variables,
                 ErrorTermContainer<ErrorTerms...> &errorTerms) {
    PROFILE(std::string(typeid(ErrorTermContainer<ErrorTerms...>).name()) +
            "_linearize");
    // loop through all error terms and linearize all of them.
    internal::static_for(
        errorTerms.tupleOfErrorTermMaps, [&](auto i, auto &errorTermMap) {
          auto linearizationFn = [&variables](auto &errorTerm) {
            errorTerm.evaluate(variables, true);
          };
          if (settings.parallelizeErrorTermLinearization) {
            // Compute the number of threads to use for execution.
            int nThreads =
                std::max(std::ceil((double)errorTermMap.size() /
                                   (double)settings.errorTermsPerThread),
                         1.0);
            LOG_INFO("Linearizing error term with {} threads", nThreads);
            // Execute
            QDVO::ParallelAlgorithms::for_each(
                QDVO::ParallelAlgorithms::PARALLEL_CPU, errorTermMap.begin(),
                errorTermMap.end(), linearizationFn, nThreads);
          } else {
            QDVO::ParallelAlgorithms::for_each(
                QDVO::ParallelAlgorithms::SEQUENTIAL, errorTermMap.begin(),
                errorTermMap.end(), linearizationFn);
          }
        });
  }

  /// Zeros all matrices in the problem.
  void setZero() {
    A.setZero();
    b_correlated.setZero();

    // iterate over all tuple elements and zero them.
    internal::static_for(B, [&](auto i, auto &array) {
      for (auto &matrix : array) {
        matrix.setZero();
      }
    });

    // iterate over all tuple elements and zero them.
    internal::static_for(D, [&](auto i, auto &array) {
      for (auto &matrix : array) {
        matrix.setZero();
      }
    });

    // Hacky way of iterating over variables.
    std::tuple<UncorrelatedVariables *...> uncorrelatedVariablesTuple;
    internal::static_for(uncorrelatedVariablesTuple, [&](auto i, auto &temp) {
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type RowVariable;

      auto &rowMap = b_uncorrelated.template getRowMap<RowVariable>();

      for (auto &block : rowMap) {
        block.setZero();
      }
    });
  }

  /**
   * Using the set of linearized error terms, Build up a linear system by
   * computing
   *
   * \f$ A = A_{0} + \sum_{i=0}^n J_{i}^{\top} \Sigma^{-1} J_{i} \rho \f$
   *
   * \f$ b = b_{0} + \sum_{i=0}^n J_{i}^{\top} \Sigma^{-1} e_{i} \rho \f$
   *
   * Where \f$ e_{i} \f$ is the residual and \f$ J_{i} \f$ is the jacobian of
   * error w.r.t to all variables.
   *
   * @return Whitened squared error. If no error terms were used, NaN is
   * returned.
   */
  template <typename GaussianPriorType>
  double buildLinearSystem(
      GaussianPriorType &prior,
      ErrorTermContainer<ErrorTerms...> &linearizedErrorTerms,
      VariableContainer<Variables...> &variables) {
    PROFILE(std::string(typeid(linearizedErrorTerms).name()) +
            "_buildLinearSystem");
    // Initialize the current problem to the prior.

    setProblemToPrior(prior, variables);

    double whitenedSqError = 0;
    int nErrorTerms = 0;

    // iterate through all error terms
    internal::static_for(
        linearizedErrorTerms.tupleOfErrorTermMaps,
        [&](auto errorTermTypeIndex, auto &errorTermMap) {
          LOG_TRACE(
              "Building problem with Error Term Type: {}",
              typeid(
                  typename std::tuple_element<errorTermTypeIndex,
                                              std::tuple<ErrorTerms...>>::type)
                  .name());
          for (auto &errorTerm : errorTermMap) {
            // Check if the linearization is valid for this error term.
            if (errorTerm.linearizationValid) {
              double sqError = errorTerm.residual.squaredNorm();
              double error = sqrt(sqError);
              double weight = lossFunction.computeWeight(error, sqError);

              whitenedSqError += sqError;
              ++nErrorTerms;

              // Iterate through all independent variables.
              internal::static_for(
                  errorTerm.variableKeys, [&](auto i, auto &outerVariableKey) {
                    typedef typename std::remove_reference<decltype(
                        outerVariableKey)>::type OuterVariableKeyType;
                    typedef typename std::remove_reference<decltype(
                        errorTerm)>::type ErrorTermType;

                    // Cache the error transformation.
                    Eigen::Matrix<
                        ScalarType,
                        OuterVariableKeyType::variable_type::dimension,
                        ErrorTermType::residual_dimension>
                        rhoJtW = (std::get<i>(errorTerm.variableJacobians)
                                      .transpose() *
                                  errorTerm.information * weight)
                                     .eval();

                    // Add to rhs.
                    addBlockToRHS(outerVariableKey,
                                  rhoJtW * -errorTerm.residual);

                    internal::static_for(
                        errorTerm.variableKeys,
                        [&](auto j, auto &innerVariableKey) {
                          // Extract the variable types from the keys.
                          // These keys must be of type
                          // VariableKey<VariableType>.
                          typedef typename std::remove_reference<decltype(
                              innerVariableKey)>::type InnerVariableKeyType;

                          constexpr bool is_inner_variable_uncorrelated =
                              internal::Is_in_tuple<
                                  typename InnerVariableKeyType::variable_type,
                                  std::tuple<UncorrelatedVariables...>>::value;
                          constexpr bool is_outer_variable_uncorrelated =
                              internal::Is_in_tuple<
                                  typename OuterVariableKeyType::variable_type,
                                  std::tuple<UncorrelatedVariables...>>::value;

                          // Compute pJtJ and pJte for this error term.
                          // The block is computed as: outer^T * inner.
                          // outer = row. inner = column.
                          // Do not compute uncorrelated x correlated.
                          if constexpr (!(is_outer_variable_uncorrelated &&
                                          !is_inner_variable_uncorrelated)) {
                            // Add to lhs.
                            addBlockToLHS(
                                outerVariableKey, innerVariableKey,
                                rhoJtW *
                                    std::get<j>(errorTerm.variableJacobians));
                          }
                        });
                  });
            } else {
              LOG_TRACE("linearization invalid for error term.");
            }
          }
        });

    if (nErrorTerms == 0) {
      LOG_WARN(
          "There are no error terms to build the problem with. Expect a nan "
          "average error.");
    }

    return whitenedSqError / nErrorTerms;
  }

  /**
   * Sets the problem exactly to the prior.
   *
   * \f$ A = A_{0} \f$
   *
   * \f$ b = b_{0} \f$
   */
  void setProblemToPrior(
      GaussianPrior<Scalar<ScalarType>, VariableGroup<Variables...>> &prior,
      VariableContainer<Variables...> &variables) {
    LOG_TRACE("Setting problem to prior.");

    PROFILE(std::string(typeid(prior).name()) + "_setProblemToPrior");

    // Zero the problem.
    setZero();

    auto &A0 = prior.A0;
    auto &b0 = prior.b0;

    // Semi hack way of iterating through all variable types
    std::tuple<Variables *...> variableTuple;

    // Iterate through A0
    internal::static_for(variableTuple, [&](auto i, auto &temp) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          RowVariable;

      auto &rowMap = A0.template getRowMap<RowVariable>();

      for (auto &keySparseBlockRowPair : rowMap) {
        // Get the key for this row.
        const VariableKey<RowVariable> &rowKey = keySparseBlockRowPair.first;
        auto &sparseBlockRow = keySparseBlockRowPair.second;

        if (variables.variableExists(rowKey)) {
          internal::static_for(variableTuple, [&](auto j, auto &temp) {
            typedef
                typename std::tuple_element<j, std::tuple<Variables...>>::type
                    ColumnVariable;

            constexpr bool row_variable_is_uncorrelated = internal::Is_in_tuple<
                RowVariable, std::tuple<UncorrelatedVariables...>>::value;
            constexpr bool column_variable_is_uncorrelated =
                internal::Is_in_tuple<
                    ColumnVariable,
                    std::tuple<UncorrelatedVariables...>>::value;

            // Skip if the variables are both uncorrelated and different.
            if constexpr (!(row_variable_is_uncorrelated &&
                            column_variable_is_uncorrelated &&
                            !std::is_same<RowVariable,
                                          ColumnVariable>::value)) {
              auto &variableMap =
                  sparseBlockRow.template getVariableMap<ColumnVariable>();

              for (auto &keyColumnMatrixPair : variableMap) {
                // Get the key for this column.
                const VariableKey<ColumnVariable> &columnKey =
                    keyColumnMatrixPair.first;
                if (variables.variableExists(columnKey)) {
                  // Add the block matrix to the problem.
                  const auto &blockMatrix = keyColumnMatrixPair.second;
                  addBlockToLHS(rowKey, columnKey, blockMatrix);
                }
              }
            }
          });
        }
      }
    });

    // Iterate through b0
    internal::static_for(variableTuple, [&](auto i, auto &temp) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          RowVariable;

      auto &priorRHSRowMap = prior.b0.template getRowMap<RowVariable>();

      for (auto it = priorRHSRowMap.begin(); it != priorRHSRowMap.end(); it++) {
        //  Get the key for this block.
        auto key =
            priorRHSRowMap.getKeyFromDataIndex(it - priorRHSRowMap.begin());
        if (variables.variableExists(key)) {
          // add the block to b.
          addBlockToRHS(key, *(it));
        }
      }
    });
  }

  /**
   * Adds a given column block to the left hand side of the problem.
   * This functions assumes that the index map has been computed.
   */
  template <typename RowVariable>
  void addBlockToRHS(
      const VariableKey<RowVariable> &rowKey,
      const Eigen::Matrix<ScalarType, RowVariable::dimension, 1> &block) {
    constexpr bool variable_is_uncorrelated =
        internal::Is_in_tuple<RowVariable,
                              std::tuple<UncorrelatedVariables...>>::value;

    if constexpr (!variable_is_uncorrelated) {
      auto &indexMap = std::get<IndexMap<RowVariable>>(variableToIndexMaps);

      auto indexIt = indexMap.at(rowKey);

      assert(indexIt != indexMap.end());

      b_correlated.template block<RowVariable::dimension, 1>(*(indexIt), 0)
          .noalias() += block;
    }
    if constexpr (variable_is_uncorrelated) {
      b_uncorrelated.getRowBlock(rowKey).noalias() += block;
    }
  }

  /**
   * Adds a given column block to the left hand side of the problem.
   * This functions assumes that the index map has been computed.
   */
  template <typename RowVariable, typename ColumnVariable>
  void addBlockToLHS(const VariableKey<RowVariable> &rowKey,
                     const VariableKey<ColumnVariable> &columnKey,
                     const Eigen::Matrix<ScalarType, RowVariable::dimension,
                                         ColumnVariable::dimension> &block) {
    // Verify that these keys are part of the variable set.
    static_assert(
        internal::Is_in_tuple<RowVariable, std::tuple<Variables...>>::value);
    static_assert(
        internal::Is_in_tuple<ColumnVariable, std::tuple<Variables...>>::value);

    constexpr bool row_variable_is_uncorrelated =
        internal::Is_in_tuple<RowVariable,
                              std::tuple<UncorrelatedVariables...>>::value;
    constexpr bool column_variable_is_uncorrelated =
        internal::Is_in_tuple<ColumnVariable,
                              std::tuple<UncorrelatedVariables...>>::value;

    // Compile time if statements used to determin where the block matrix should
    // be added to.
    if constexpr (row_variable_is_uncorrelated &&
                  column_variable_is_uncorrelated) {
      // Add this block to D.
      // These keys should be the same type.
      static_assert(std::is_same<RowVariable, ColumnVariable>::value);
      // These keys should be equal.
      assert(rowKey == columnKey);

      // Add the block to D.
      auto &slotArray = std::get<DVector<RowVariable>>(D);
      auto it = slotArray.at(rowKey);

      // The matrix should exist.
      assert(it != slotArray.end());

      auto &lhs = *(it);
      lhs.noalias() += block;
    }

    if constexpr (!row_variable_is_uncorrelated &&
                  !column_variable_is_uncorrelated) {
      // Add this block to A.
      auto rowIdxIt =
          std::get<IndexMap<RowVariable>>(variableToIndexMaps).at(rowKey);
      auto colIdxIt =
          std::get<IndexMap<ColumnVariable>>(variableToIndexMaps).at(columnKey);
      assert(rowIdxIt !=
             std::get<IndexMap<RowVariable>>(variableToIndexMaps).end());
      assert(colIdxIt !=
             std::get<IndexMap<ColumnVariable>>(variableToIndexMaps).end());

      const size_t rowIdx = *(rowIdxIt);
      const size_t colIdx = *(colIdxIt);

      A.template block<RowVariable::dimension, ColumnVariable::dimension>(
           rowIdx, colIdx)
          .noalias() += block;
    }

    if constexpr (!row_variable_is_uncorrelated &&
                  column_variable_is_uncorrelated) {
      // Add this block to B.
      auto rowIdxIt =
          std::get<IndexMap<RowVariable>>(variableToIndexMaps).at(rowKey);
      assert(rowIdxIt !=
             std::get<IndexMap<RowVariable>>(variableToIndexMaps).end());
      const size_t rowIdx = *(rowIdxIt);

      auto &slotArray = std::get<BVector<ColumnVariable>>(B);
      auto it = slotArray.at(columnKey);

      assert(it != slotArray.end());
      auto &bMatrix = *(it);
      bMatrix
          .template block<RowVariable::dimension, ColumnVariable::dimension>(
              rowIdx, 0)
          .noalias() += block;
    }
  }

  /// Precomputes the map bewteen correlated variable keys and their index in A.
  void precomputeIndexMapAndResizeMatrices(
      VariableContainer<Variables...> &variables) {
    dimensionOfA = 0;

    /// compute the index map starting with only the correlated variables.
    internal::static_for(variables.tupleOfVariableMaps, [&](auto i,
                                                            auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      // Only set the dimensions if this variable is not part of the
      // uncorrelated set.
      if constexpr (!(internal::Is_in_tuple<
                        ThisVariable,
                        std::tuple<UncorrelatedVariables...>>::value)) {
        // Clear the index map before.
        std::get<IndexMap<ThisVariable>>(variableToIndexMaps).clear();

        for (size_t idx = 0; idx < variableMap.size(); ++idx) {
          auto key = variableMap.getKeyFromDataIndex(idx);
          assert(variables.variableExists(key));

          std::get<IndexMap<ThisVariable>>(variableToIndexMaps)
              .insert(key, dimensionOfA);

          dimensionOfA += ThisVariable::dimension;
        }

        assert(std::get<IndexMap<ThisVariable>>(variableToIndexMaps).size() ==
               variableMap.size());
      }
    });

    totalDimension = dimensionOfA;

    /// Compute the remaining variables in the index map (uncorrelated set).
    internal::static_for(variables.tupleOfVariableMaps, [&](auto i,
                                                            auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      // Only set the dimensions if this variable is part of the
      // uncorrelated set.
      if constexpr ((internal::Is_in_tuple<
                        ThisVariable,
                        std::tuple<UncorrelatedVariables...>>::value)) {
        // Clear the index map before.
        std::get<IndexMap<ThisVariable>>(variableToIndexMaps).clear();

        for (size_t idx = 0; idx < variableMap.size(); ++idx) {
          auto key = variableMap.getKeyFromDataIndex(idx);
          assert(variables.variableExists(key));

          std::get<IndexMap<ThisVariable>>(variableToIndexMaps)
              .insert(key, totalDimension);

          totalDimension += ThisVariable::dimension;
        }

        assert(std::get<IndexMap<ThisVariable>>(variableToIndexMaps).size() ==
               variableMap.size());
      }
    });

    // Resize the dense portion of dx.
    dx.resize(totalDimension, 1);

    // Resize A if necessary.
    assert(A.rows() == A.cols());
    // if (A.rows() < dimensionOfA) {
    A.resize(dimensionOfA, dimensionOfA);
    // }

    // Resize b_correlated if necessary
    // if (b_correlated.rows() < dimensionOfA) {
    b_correlated.resize(dimensionOfA, 1);
    //}

    // Resize the matrices of B if necessary
    internal::static_for(B, [&](auto i, auto &array) {
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type ThisVariable;
      for (auto &matrix : array) {
        // if (matrix.rows() < dimensionOfA) {
        matrix.resize(dimensionOfA, ThisVariable::dimension);
        //}
      }
    });

    // Resize the matrices of negativeBDinv if necessary
    internal::static_for(negativeBDinv, [&](auto i, auto &array) {
      typedef typename std::tuple_element<
          i, std::tuple<UncorrelatedVariables...>>::type ThisVariable;
      for (auto &matrix : array) {
        // if (matrix.rows() < dimensionOfA) {
        matrix.resize(dimensionOfA, ThisVariable::dimension);
        //}
      }
    });
  }

  /// Ensures that the slot arrays stored in the solver are in sync with the
  /// current variable set.
  void addNewVariablesToSlotArrays(VariableContainer<Variables...> &variables) {
    // Insert variables if they do not exist.
    internal::static_for(variables.tupleOfVariableMaps, [&](auto i,
                                                            auto &variableMap) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      Eigen::Matrix<ScalarType, ThisVariable::dimension, 1> zeroRHSMatrix =
          RHSBlockVector::template MatrixBlock<ThisVariable>::Zero();

      for (size_t idx = 0; idx < variableMap.size(); ++idx) {
        auto key = variableMap.getKeyFromDataIndex(idx);
        assert(variables.variableExists(key));
        // Only do this for uncorrelated variables.
        if constexpr (internal::Is_in_tuple<
                          ThisVariable,
                          std::tuple<UncorrelatedVariables...>>::value) {
          // Check if the key exists in B
          if (std::get<BVector<ThisVariable>>(B).at(key) ==
              std::get<BVector<ThisVariable>>(B).end()) {
            std::get<BVector<ThisVariable>>(B).insert(
                key,
                Eigen::Matrix<ScalarType, Eigen::Dynamic,
                              ThisVariable::dimension>::Zero(dimensionOfA));
          }

          // Check if the key exists in negativeBDinv
          if (std::get<BVector<ThisVariable>>(negativeBDinv).at(key) ==
              std::get<BVector<ThisVariable>>(negativeBDinv).end()) {
            std::get<BVector<ThisVariable>>(negativeBDinv)
                .insert(
                    key,
                    Eigen::Matrix<ScalarType, Eigen::Dynamic,
                                  ThisVariable::dimension>::Zero(dimensionOfA));
          }

          // Check if the key exists in D
          if (std::get<DVector<ThisVariable>>(D).at(key) ==
              std::get<DVector<ThisVariable>>(D).end()) {
            std::get<DVector<ThisVariable>>(D).insert(
                key, DBlock<ThisVariable>::Zero());
          }

          // Check if the key exists in b_uncorrelated
          if (!b_uncorrelated.blockExists(key)) {
            b_uncorrelated.addRowBlock(key, zeroRHSMatrix);
          }
        }

        // Add an elements to the dx block vector.
        if (!dxBlockVector.blockExists(key)) {
          dxBlockVector.addRowBlock(key, zeroRHSMatrix);
        }
      }
    });
  }

  /// Ensures that the internal slot arrays do not have excess variables stored
  /// in them.
  void removeOldVariablesFromSlotArrays(
      VariableContainer<Variables...> &variables) {
    std::tuple<std::vector<VariableKey<Variables>>...> keysToErase;

    // Find keys in the Dx block vector which are not in the variable container.
    std::tuple<Variables *...> variableTuple;
    internal::static_for(variableTuple, [&](auto i, auto &donotuseme) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;
      auto &variableMap = dxBlockVector.template getRowMap<ThisVariable>();

      for (auto it = variableMap.begin(); it != variableMap.end(); it++) {
        auto key = variableMap.getKeyFromDataIndex(it - variableMap.begin());

        if (!variables.variableExists(key)) {
          std::get<std::vector<decltype(key)>>(keysToErase).push_back(key);
        }
      }
      LOG_TRACE("Found {} remove variables of type {}",
                std::get<i>(keysToErase).size(),
                typeid(std::get<i>(variableTuple)).name());
    });

    // Erase all keys which are not in the variable container, but exist in the
    // solver.
    internal::static_for(keysToErase, [&](auto i, auto &keyVector) {
      typedef typename std::tuple_element<i, std::tuple<Variables...>>::type
          ThisVariable;

      if constexpr (internal::Is_in_tuple<
                        ThisVariable,
                        std::tuple<UncorrelatedVariables...>>::value) {
        // Erase D
        for (const auto &key : keyVector) {
          std::get<DVector<ThisVariable>>(D).erase(key);
        }

        // Erase B
        for (const auto &key : keyVector) {
          std::get<BVector<ThisVariable>>(B).erase(key);
        }

        // Erase -negBDinv
        for (const auto &key : keyVector) {
          std::get<BVector<ThisVariable>>(negativeBDinv).erase(key);
        }

        // Erase b_uncorr
        for (const auto &key : keyVector) {
          b_uncorrelated.removeRowBlock(key);
        }

        // TODO Find a way to remove these variables properly.
        // Erase dx block vector
        // for (const auto &key : keyVector) {
        // dxBlockVector.removeRowBlock(key);
        //}
      }
    });
  }

  /// Updates the pointers to variables inside the error terms.
  void updateVariablePointers(VariableContainer<Variables...> &variables,
                              ErrorTermContainer<ErrorTerms...> &errorTerms) {
    // iterate over all error term maps
    internal::static_for(errorTerms.tupleOfErrorTermMaps,
                         [&](auto i, auto &errorTermMap) {
                           for (auto &errorTerm : errorTermMap) {
                             errorTerm.updateVariablePointers(variables);
                           }
                         });
  }
};

}  // namespace ArgMin
