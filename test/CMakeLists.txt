add_executable(ArgMinTests
  TestArgMin.cpp
  TestArgMinExampleProblem.cpp
  TestSlotMap.cpp
  TestSlotArray.cpp
  TestParallelAlgorithms.cpp
)

target_link_libraries(ArgMinTests PRIVATE
  ArgMin::ArgMin
  GTest::gtest
  GTest::gtest_main
)

# Ensure MSVC uses the same runtime library as GoogleTest
if(MSVC)
  set_target_properties(ArgMinTests PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )
endif()

include(GoogleTest)
gtest_discover_tests(ArgMinTests)
