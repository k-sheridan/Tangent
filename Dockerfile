# Multi-stage Dockerfile for ArgMin library
# Provides isolated build and test environment

# Build stage
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies and ArgMin runtime dependencies
# Note: libsophus-dev not available in Ubuntu 22.04, will use FetchContent
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    libssl-dev \
    libeigen3-dev \
    libspdlog-dev \
    libgtest-dev \
    libbenchmark-dev \
    && rm -rf /var/lib/apt/lists/*

# Install modern C++20 compiler (GCC 12)
RUN apt-get update && apt-get install -y \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /argmin

# Copy source files
COPY . .

# Configure CMake with all options enabled
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DARGMIN_BUILD_TESTS=ON \
    -DARGMIN_BUILD_BENCHMARKS=ON \
    -DARGMIN_USE_SPDLOG=OFF

# Build everything
RUN cmake --build build --parallel $(nproc)

# Test stage
FROM builder AS test
WORKDIR /argmin/build
CMD ["ctest", "--output-on-failure", "--verbose"]

# Benchmark stage
FROM builder AS benchmark
WORKDIR /argmin/build
CMD ["./bench/ArgMinBenchmarks"]

# Development stage - includes all tools for interactive use
FROM builder AS dev

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /argmin

# Default to bash for interactive use
CMD ["/bin/bash"]
