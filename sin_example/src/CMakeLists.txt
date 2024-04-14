add_executable(
        example
        sin_example.cpp
)

set_target_properties(
        example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(
        example PUBLIC
        "${PERCEPTRON_INCLUDE_DIRS}"
)

target_link_libraries(example PUBLIC tbb boost_iostreams boost_system boost_filesystem)