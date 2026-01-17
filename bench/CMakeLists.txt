add_executable(ArgMinBenchmarks
  MainBenchmark.cpp
  SlotMapBenchmark.cpp
  SparseBlockMatrixBenchmark.cpp
)

target_link_libraries(ArgMinBenchmarks PRIVATE
  ArgMin::ArgMin
  benchmark::benchmark
  benchmark::benchmark_main
)

# Ensure MSVC uses the same runtime library
if(MSVC)
  set_target_properties(ArgMinBenchmarks PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )
endif()
