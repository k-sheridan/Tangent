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

# Apply sanitizer flags if enabled
if(ARGMIN_ENABLE_ASAN)
  target_compile_options(ArgMinBenchmarks PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
  target_link_options(ArgMinBenchmarks PRIVATE -fsanitize=address)
endif()

if(ARGMIN_ENABLE_TSAN)
  target_compile_options(ArgMinBenchmarks PRIVATE -fsanitize=thread -g)
  target_link_options(ArgMinBenchmarks PRIVATE -fsanitize=thread)
endif()

if(ARGMIN_ENABLE_UBSAN)
  target_compile_options(ArgMinBenchmarks PRIVATE -fsanitize=undefined -fno-omit-frame-pointer -g)
  target_link_options(ArgMinBenchmarks PRIVATE -fsanitize=undefined)
endif()

# Ensure MSVC uses the same runtime library
if(MSVC)
  set_target_properties(ArgMinBenchmarks PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )
endif()
