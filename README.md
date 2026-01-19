# ArgMin

[![Tests](https://github.com/k-sheridan/ArgMin/actions/workflows/test.yml/badge.svg)](https://github.com/k-sheridan/ArgMin/actions/workflows/test.yml)
[![Benchmarks](https://github.com/k-sheridan/ArgMin/actions/workflows/benchmark.yml/badge.svg)](https://github.com/k-sheridan/ArgMin/actions/workflows/benchmark.yml)

**Header-only generic optimizer for manifold-based nonlinear least squares**

Originally designed for sliding window estimation in visual SLAM and odometry.

## Features

- SE3/SO3 manifold optimization with Lie algebra.
- Built in marginalization support through Sparese Gaussian Prior.
- Sparse Schur complement solver for exploiting sparsity in uncorrelated variables.
- Cache-friendly SlotMap data structures (O(1) operations)
- Compile-time type safety with template metaprogramming
- Optional parallel algorithms for multi-threaded optimization

## Installation

**Add to your CMake project:**
```cmake
include(FetchContent)
FetchContent_Declare(argmin
  GIT_REPOSITORY https://github.com/k-sheridan/ArgMin.git
  GIT_TAG main
)
FetchContent_MakeAvailable(argmin)
target_link_libraries(your_target PRIVATE ArgMin::ArgMin)
```

## Usage

Refer to [test/TestArgMinExampleProblem.cpp](test/TestArgMinExampleProblem.cpp) for an example on how to use.

## Testing & Benchmarking

**Docker:**
```bash
docker-compose up test       # Run All ArgMin tests
docker-compose up benchmark  # Run performance benchmarks
```

## Requirements

- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+
- Eigen 3.4+ (auto-fetched if not found)
- Sophus 1.22+ (auto-fetched if not found)

## Notes 

ArgMin was originally developed as part of the [QDVO (Quasi-Direct Visual Odometry)](https://github.com/k-sheridan/qdvo) project.
