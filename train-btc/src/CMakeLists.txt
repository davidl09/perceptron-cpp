add_executable(
        train-btc
        main.cpp
)

set_target_properties(
        train-btc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(
        train-btc PUBLIC
        "${PERCEPTRON_INCLUDE_DIRS}"
)

target_link_libraries(
        train-btc PUBLIC
        boost_iostreams
        boost_filesystem
)
install(TARGETS train-btc RUNTIME DESTINATION "/usr/local/bin")