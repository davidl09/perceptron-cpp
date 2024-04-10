cmake_minimum_required(VERSION 3.27)
project(perceptron)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(perceptron main.cpp)

target_include_directories(perceptron PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perceptron PUBLIC tbb boost_iostreams boost_system boost_filesystem)
