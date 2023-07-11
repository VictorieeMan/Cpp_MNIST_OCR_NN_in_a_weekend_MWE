# CMakeList.txt : CMake project for my_first_cpp_NN_experiment, include source and define
# project specific logic here.
#

# Add source to this project's executable.
file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
set(SOURCE_FILES ${CPP_FILES} ${H_FILES})
add_executable(main ${SOURCE_FILES} "model.cpp" "model.h" "MNIST.h" "MNIST.cpp" "FFNode.h" "FFNode.cpp" "CCELossNode.h" "CCELossNode.cpp" "GDOptimizer.h" "GDOptimizer.cpp")


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
