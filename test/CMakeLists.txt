add_executable(ArgMinTests
  TestArgMin.cpp
  TestArgMinExampleProblem.cpp
  TestSlotMap.cpp
  TestSlotArray.cpp
  TestParallelAlgorithms.cpp
  TestEdgeCases.cpp
  TestVariableTypes.cpp
  TestMarginalization.cpp
  TestSSEOptimizer.cpp
)

target_link_libraries(ArgMinTests PRIVATE
  ArgMin::ArgMin
  GTest::gtest
  GTest::gtest_main
)

# Apply sanitizer flags if enabled
if(ARGMIN_ENABLE_ASAN)
  target_compile_options(ArgMinTests PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
  target_link_options(ArgMinTests PRIVATE -fsanitize=address)
endif()

if(ARGMIN_ENABLE_TSAN)
  target_compile_options(ArgMinTests PRIVATE -fsanitize=thread -g)
  target_link_options(ArgMinTests PRIVATE -fsanitize=thread)
endif()

if(ARGMIN_ENABLE_UBSAN)
  target_compile_options(ArgMinTests PRIVATE -fsanitize=undefined -fno-omit-frame-pointer -g)
  target_link_options(ArgMinTests PRIVATE -fsanitize=undefined)
endif()

# Ensure MSVC uses the same runtime library as GoogleTest
if(MSVC)
  set_target_properties(ArgMinTests PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )
endif()

include(GoogleTest)
gtest_discover_tests(ArgMinTests)
