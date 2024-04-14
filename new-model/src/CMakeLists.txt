add_executable(
        new-model
        main.cpp
)

set_target_properties(
        new-model PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(
        new-model PUBLIC
        "${PERCEPTRON_INCLUDE_DIRS}"
)

install(TARGETS new-model RUNTIME DESTINATION "/usr/local/bin")