include(FetchContent)

# Eigen3 (required)
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found, fetching from source...")
  message(STATUS "Tip: Install system packages for faster builds:")
  message(STATUS "  Ubuntu/Debian: sudo apt install libeigen3-dev")
  message(STATUS "  macOS: brew install eigen")
  FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(eigen)
else()
  message(STATUS "Found Eigen3: ${Eigen3_DIR}")
endif()

# Sophus (required)
find_package(Sophus QUIET)
if(NOT Sophus_FOUND)
  message(STATUS "Sophus not found, fetching from source...")
  message(STATUS "Tip: Ubuntu 22.04+: sudo apt install libsophus-dev")
  message(STATUS "     macOS: brew install sophus")
  FetchContent_Declare(sophus
    GIT_REPOSITORY https://github.com/strasdat/Sophus.git
    GIT_TAG 1.22.10
    GIT_SHALLOW TRUE
  )
  set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(sophus)
else()
  message(STATUS "Found Sophus: ${Sophus_DIR}")
endif()

# spdlog (optional)
if(ARGMIN_USE_SPDLOG)
  find_package(spdlog QUIET)
  if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found, fetching from source...")
    message(STATUS "Tip: Ubuntu/Debian: sudo apt install libspdlog-dev")
    message(STATUS "     macOS: brew install spdlog")
    FetchContent_Declare(spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.13.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(spdlog)
  else()
    message(STATUS "Found spdlog: ${spdlog_DIR}")
  endif()
endif()

# GoogleTest (for tests only)
if(ARGMIN_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found, fetching from source...")
    message(STATUS "Tip: Ubuntu/Debian: sudo apt install libgtest-dev")
    message(STATUS "     macOS: brew install googletest")
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  else()
    message(STATUS "Found GoogleTest: ${GTest_DIR}")
  endif()
endif()

# Google Benchmark (for benchmarks only)
if(ARGMIN_BUILD_BENCHMARKS)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from source...")
    message(STATUS "Tip: Ubuntu 22.04+: sudo apt install libbenchmark-dev")
    message(STATUS "     macOS: brew install google-benchmark")
    FetchContent_Declare(googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.8.3
      GIT_SHALLOW TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
  else()
    message(STATUS "Found Google Benchmark: ${benchmark_DIR}")
  endif()
endif()
