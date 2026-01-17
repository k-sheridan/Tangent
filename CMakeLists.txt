cmake_minimum_required(VERSION 3.20)
project(ArgMin VERSION 1.0.0 LANGUAGES CXX)

# Project configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ARGMIN_BUILD_TESTS "Build tests" ON)
option(ARGMIN_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ARGMIN_USE_SPDLOG "Enable spdlog logging" OFF)

# Create header-only library target
add_library(ArgMin INTERFACE)
add_library(ArgMin::ArgMin ALIAS ArgMin)

target_include_directories(ArgMin INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(ArgMin INTERFACE cxx_std_20)

# Dependencies
include(cmake/Dependencies.cmake)

# Link dependencies (use $<BUILD_INTERFACE> to avoid export issues with FetchContent)
target_link_libraries(ArgMin INTERFACE
  $<BUILD_INTERFACE:Eigen3::Eigen>
  $<BUILD_INTERFACE:Sophus::Sophus>
)

if(ARGMIN_USE_SPDLOG)
  target_compile_definitions(ArgMin INTERFACE ARGMIN_USE_SPDLOG)
  target_link_libraries(ArgMin INTERFACE $<BUILD_INTERFACE:spdlog::spdlog>)
endif()

# Testing
if(ARGMIN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(ARGMIN_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ArgMin EXPORT ArgMinTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ArgMinTargets
  FILE ArgMinTargets.cmake
  NAMESPACE ArgMin::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ArgMin
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ArgMinConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "cmake/ArgMinConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ArgMinConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ArgMin
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ArgMinConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ArgMinConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ArgMin
)
