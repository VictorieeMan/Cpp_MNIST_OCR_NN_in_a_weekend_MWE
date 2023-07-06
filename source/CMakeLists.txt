# CMakeList.txt : CMake project for my_first_cpp_NN_experiment, include source and define
# project specific logic here.
#

# Add source to this project's executable.
file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
set(SOURCE_FILES ${CPP_FILES} ${H_FILES})
add_executable(my_executable ${SOURCE_FILES})


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET my_first_cpp_NN_experiment PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
