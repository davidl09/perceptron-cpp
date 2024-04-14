cmake_minimum_required(VERSION 3.27)
project(perceptron)

set(CMAKE_CXX_STANDARD 23)
set(PERCEPTRON_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "${PERCEPTRON_INCLUDE_DIRS}")

add_subdirectory(sin_example)
add_subdirectory(new-model)