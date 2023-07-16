# CMakeList.txt : Cmake file to include the MNIST data set files in the build

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()

#[[
# Copy the MNIST data set files to the build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/t10k-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/t10k-images.idx3-ubyte COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/t10k-labels.idx1-ubyte ${CMAKE_CURRENT_BINARY_DIR}/t10k-labels.idx1-ubyte COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/train-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-images.idx3-ubyte COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/train-labels.idx1-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-labels.idx1-ubyte COPYONLY)
]]