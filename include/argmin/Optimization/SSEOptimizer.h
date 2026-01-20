#pragma once

#include "argmin/Types/MetaHelpers.h"
#include "argmin/Containers/Key.h"
#include "argmin/Optimization/OptimizerContainers.h"

#include <limits>
#include <vector>

namespace ArgMin
{

template <typename...>
class SSEOptimizer;

template <typename ScalarType, typename Solver, typename Prior, typename Marginalizer, typename... Variables, typename... ErrorTerms>
class SSEOptimizer<Scalar<ScalarType>, Solver, Prior, Marginalizer, VariableGroup<Variables...>, ErrorTermGroup<ErrorTerms...>>
{
public:
    // ========================================================================
    // Type Aliases
    // ========================================================================
    using ScalarT = ScalarType;
    using VariableContainerT = VariableContainer<Variables...>;
    using ErrorTermContainerT = ErrorTermContainer<ErrorTerms...>;
    using PriorT = Prior;
    using SolverT = Solver;
    using MarginalizerT = Marginalizer;

    // ========================================================================
    // Result Structs
    // ========================================================================

    /// Result of an optimization run.
    struct OptimizationResult {
        std::vector<ScalarType> errorHistory;  ///< Error at each iteration
        size_t iterations = 0;                 ///< Total iterations performed
        ScalarType initialError = 0;           ///< Error before optimization
        ScalarType finalError = 0;             ///< Error after optimization
        bool converged = false;                ///< Whether convergence criteria were met
        bool errorDecreased = false;           ///< Whether final error < initial error
    };

    /// Result of a marginalization operation.
    struct MarginalizationResult {
        bool success = false;                  ///< Whether marginalization succeeded
        size_t errorTermsRemoved = 0;          ///< Number of error terms removed
    };

    // ========================================================================
    // Settings
    // ========================================================================

    /// Configuration settings for the optimizer.
    struct Settings {
        // Solver settings (mirrored from PSDSchurSolver)
        ScalarType initialLambda = 1e3;
        ScalarType lambdaReductionMultiplier = 10;
        int maximumIterations = 50;
        bool stopAfterErrorIncrease = false;
        bool parallelizeErrorTermLinearization = false;
        ScalarType errorTermsPerThread = 1000;
        ScalarType updateThreshold = 0;
        ScalarType errorDeltaThreshold = 0;

        // Marginalization settings
        ScalarType marginalizationResidualThreshold = std::numeric_limits<ScalarType>::max();
    };

    /// Optimizer settings.
    Settings settings;

    // ========================================================================
    // Constructor
    // ========================================================================

    /// Default constructor. Uses a default-constructed solver.
    /// Note: This may not work if the Solver type requires initialization parameters.
    SSEOptimizer() = default;

    /// Constructs the optimizer with a pre-configured solver.
    explicit SSEOptimizer(Solver solverInstance) : solver(std::move(solverInstance)) {}

    // ========================================================================
    // Variable Management
    // ========================================================================

    /// Inserts a variable into the optimizer with default weak prior.
    template <typename VariableType>
    VariableKey<VariableType> addVariable(VariableType &variable)
    {
        auto key = variables.template getVariableMap<VariableType>().insert(variable);
        prior.addVariable(key);
        return key;
    }

    /// Inserts a variable into the optimizer with a custom prior information matrix.
    template <typename VariableType>
    VariableKey<VariableType> addVariable(
        VariableType &variable,
        const Eigen::Matrix<ScalarType, VariableType::dimension, VariableType::dimension> &informationMatrix)
    {
        auto key = variables.template getVariableMap<VariableType>().insert(variable);
        prior.addVariable(key, informationMatrix);
        return key;
    }

    /// Updates the prior information matrix for an existing variable.
    template <typename VariableType>
    void setVariablePrior(
        const VariableKey<VariableType> &key,
        const Eigen::Matrix<ScalarType, VariableType::dimension, VariableType::dimension> &informationMatrix)
    {
        prior.A0.setBlock(key, key, informationMatrix);
    }

    /// Removes a variable from the problem without marginalizing it.
    template <typename VariableType>
    void removeVariable(VariableKey<VariableType> &key)
    {
        variables.template getVariableMap<VariableType>().erase(key);
    }

    /// Gets a reference to a variable by key.
    template <typename VariableType>
    VariableType& getVariable(const VariableKey<VariableType> &key)
    {
        return variables.at(key);
    }

    /// Gets a const reference to a variable by key.
    template <typename VariableType>
    const VariableType& getVariable(const VariableKey<VariableType> &key) const
    {
        return variables.at(key);
    }

    /// Checks if a variable exists in the optimizer.
    template <typename VariableType>
    bool hasVariable(const VariableKey<VariableType> &key)
    {
        return variables.variableExists(key);
    }

    /// Returns the number of variables of a specific type.
    template <typename VariableType>
    size_t variableCount()
    {
        return variables.template getVariableMap<VariableType>().size();
    }

    /// Returns all keys for variables of a specific type.
    template <typename VariableType>
    std::vector<VariableKey<VariableType>> getVariableKeys()
    {
        std::vector<VariableKey<VariableType>> keys;
        auto& map = variables.template getVariableMap<VariableType>();
        keys.reserve(map.size());
        for (size_t i = 0; i < map.size(); ++i) {
            keys.push_back(map.getKeyFromDataIndex(i));
        }
        return keys;
    }

    /// Returns the total dimension of all variables.
    size_t totalDimension()
    {
        return variables.totalDimensions();
    }

    // ========================================================================
    // Error Term Management
    // ========================================================================

    /// Adds an error term to the problem.
    template <typename ErrorTermType>
    ErrorTermKey<ErrorTermType> addErrorTerm(const ErrorTermType &errorTerm)
    {
        return errorTerms.insert(errorTerm);
    }

    /// Removes an error term from the problem.
    template <typename ErrorTermType>
    void removeErrorTerm(const ErrorTermKey<ErrorTermType> &key)
    {
        errorTerms.erase(key);
    }

    /// Checks if an error term exists.
    template <typename ErrorTermType>
    bool hasErrorTerm(const ErrorTermKey<ErrorTermType> &key)
    {
        return errorTerms.template getErrorTermMap<ErrorTermType>().at(key) !=
               errorTerms.template getErrorTermMap<ErrorTermType>().end();
    }

    /// Returns the number of error terms of a specific type.
    template <typename ErrorTermType>
    size_t errorTermCount()
    {
        return errorTerms.template getErrorTermMap<ErrorTermType>().size();
    }

    /// Clears all error terms from the problem.
    void clearErrorTerms()
    {
        errorTerms.clear();
    }

    // ========================================================================
    // Optimization
    // ========================================================================

    /// Performs Levenberg-Marquardt optimization using current settings.
    OptimizationResult optimize()
    {
        OptimizationResult result;

        // Clean prior - remove variables that no longer exist
        prior.removeUnsedVariables(variables);

        // Apply settings to solver
        solver.settings.initialLambda = settings.initialLambda;
        solver.settings.lambdaReductionMultiplier = settings.lambdaReductionMultiplier;
        solver.settings.maximumIterations = settings.maximumIterations;
        solver.settings.stopAfterErrorIncrease = settings.stopAfterErrorIncrease;
        solver.settings.parallelizeErrorTermLinearization = settings.parallelizeErrorTermLinearization;
        solver.settings.errorTermsPerThread = settings.errorTermsPerThread;
        solver.settings.updateThreshold = settings.updateThreshold;
        solver.settings.errorDeltaThreshold = settings.errorDeltaThreshold;

        // Run solver
        auto solverResult = solver.solveLevenbergMarquardt(variables, errorTerms, prior);

        // Package results
        result.errorHistory = std::move(solverResult.whitenedSqError);
        result.iterations = result.errorHistory.size();

        if (result.iterations > 0) {
            result.initialError = result.errorHistory.front();
            result.finalError = result.errorHistory.back();
            result.errorDecreased = result.finalError < result.initialError;
            result.converged = result.iterations < static_cast<size_t>(settings.maximumIterations);
        } else {
            result.initialError = std::numeric_limits<ScalarType>::quiet_NaN();
            result.finalError = std::numeric_limits<ScalarType>::quiet_NaN();
            result.errorDecreased = false;
            result.converged = false;
        }

        return result;
    }

    // ========================================================================
    // Marginalization
    // ========================================================================

    /// Marginalizes a variable, removing associated error terms and updating the prior.
    /// Uses default residual threshold from settings.
    template <typename VariableType>
    MarginalizationResult marginalizeVariable(VariableKey<VariableType> &key)
    {
        return marginalizeVariable(key, VariableGroup<>(), settings.marginalizationResidualThreshold);
    }

    /// Marginalizes a variable while ignoring correlations with specific variable types.
    /// This is essential for preserving sparsity (e.g., ignore InverseDepth when marginalizing SE3).
    template <typename VariableType, typename... IgnoredVariables>
    MarginalizationResult marginalizeVariable(
        VariableKey<VariableType> &key,
        VariableGroup<IgnoredVariables...> ignoredTypes)
    {
        return marginalizeVariable(key, ignoredTypes, settings.marginalizationResidualThreshold);
    }

    /// Marginalizes a variable with custom residual threshold for outlier rejection.
    template <typename VariableType, typename... IgnoredVariables>
    MarginalizationResult marginalizeVariable(
        VariableKey<VariableType> &key,
        VariableGroup<IgnoredVariables...> ignoredTypes,
        ScalarType residualThreshold)
    {
        MarginalizationResult result;

        // Perform marginalization via the Marginalizer
        bool success = marginalizer.marginalizeVariable(
            key, prior, errorTerms, ignoredTypes, residualThreshold);

        if (success) {
            // Remove the variable from the container
            variables.template getVariableMap<VariableType>().erase(key);
        }

        result.success = success;
        return result;
    }

    // ========================================================================
    // Direct Access (for advanced use cases)
    // ========================================================================

    /// Gets reference to internal solver.
    Solver& getSolver() { return solver; }
    const Solver& getSolver() const { return solver; }

    /// Gets reference to internal prior.
    Prior& getPrior() { return prior; }
    const Prior& getPrior() const { return prior; }

    /// Gets reference to internal marginalizer.
    Marginalizer& getMarginalizer() { return marginalizer; }
    const Marginalizer& getMarginalizer() const { return marginalizer; }

    /// Gets reference to variable container.
    VariableContainerT& getVariables() { return variables; }
    const VariableContainerT& getVariables() const { return variables; }

    /// Gets reference to error term container.
    ErrorTermContainerT& getErrorTerms() { return errorTerms; }
    const ErrorTermContainerT& getErrorTerms() const { return errorTerms; }

    // ========================================================================
    // Members
    // ========================================================================

    /// Variable container.
    VariableContainer<Variables...> variables;

    /// Error term container.
    ErrorTermContainer<ErrorTerms...> errorTerms;

    /// Prior
    Prior prior;

    /// Marginalizer
    Marginalizer marginalizer;

    /// Solver
    Solver solver;
};

} // namespace ArgMin
