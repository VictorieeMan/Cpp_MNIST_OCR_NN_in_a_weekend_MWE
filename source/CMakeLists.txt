# CMakeList.txt : CMake project for my_first_cpp_NN_experiment, include source and define
# project specific logic here.
#

add_executable(main
	"main.cpp"
    "model.h"
    "model.cpp"
    "MNIST.h"
    "MNIST.cpp"
    "FFNode.h"
    "FFNode.cpp"
    "CCELossNode.h"
    "CCELossNode.cpp"
    "GDOptimizer.h"
    "GDOptimizer.cpp"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()

# Set the output folder where your program will be created
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Add source to this project's executable.
#[[ NOTE: not recommended to use GLOB to collect source files,
because it will not detect new files added to the project and
it will avoid adding changes in dependencies to git.
Better to list all files explicitly.
]]
#[[	Old solution:
file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
set(SOURCE_FILES ${CPP_FILES} ${H_FILES})
add_executable(main ${SOURCE_FILES})

New solution implemented above ^^
]]
