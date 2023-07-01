# CMakeList.txt : CMake project for my_first_cpp_NN_experiment, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (my_first_cpp_NN_experiment "my_first_cpp_NN_experiment.cpp" "my_first_cpp_NN_experiment.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET my_first_cpp_NN_experiment PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
