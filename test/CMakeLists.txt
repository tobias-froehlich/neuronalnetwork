find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable( Test
  test.cpp
  ../src/utils.cpp
  ../src/Neuron.cpp
  ../src/Network.cpp
  ../src/Database.cpp
  ../src/Trainer.cpp
)

target_link_libraries(Test ${GTEST_LIBRARIES} pthread)
