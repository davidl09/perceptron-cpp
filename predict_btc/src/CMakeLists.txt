add_executable(
        btc-pred
        main.cpp
)

set_target_properties(
        btc-pred PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(
        btc-pred PUBLIC
        "${PERCEPTRON_INCLUDE_DIRS}"
)