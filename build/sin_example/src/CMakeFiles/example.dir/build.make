# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/davidl09/CLionProjects/perceptron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/davidl09/CLionProjects/perceptron/build

# Include any dependencies generated for this target.
include sin_example/src/CMakeFiles/example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include sin_example/src/CMakeFiles/example.dir/compiler_depend.make

# Include the progress variables for this target.
include sin_example/src/CMakeFiles/example.dir/progress.make

# Include the compile flags for this target's objects.
include sin_example/src/CMakeFiles/example.dir/flags.make

sin_example/src/CMakeFiles/example.dir/sin_example.cpp.o: sin_example/src/CMakeFiles/example.dir/flags.make
sin_example/src/CMakeFiles/example.dir/sin_example.cpp.o: /home/davidl09/CLionProjects/perceptron/sin_example/src/sin_example.cpp
sin_example/src/CMakeFiles/example.dir/sin_example.cpp.o: sin_example/src/CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/davidl09/CLionProjects/perceptron/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sin_example/src/CMakeFiles/example.dir/sin_example.cpp.o"
	cd /home/davidl09/CLionProjects/perceptron/build/sin_example/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT sin_example/src/CMakeFiles/example.dir/sin_example.cpp.o -MF CMakeFiles/example.dir/sin_example.cpp.o.d -o CMakeFiles/example.dir/sin_example.cpp.o -c /home/davidl09/CLionProjects/perceptron/sin_example/src/sin_example.cpp

sin_example/src/CMakeFiles/example.dir/sin_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/example.dir/sin_example.cpp.i"
	cd /home/davidl09/CLionProjects/perceptron/build/sin_example/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/davidl09/CLionProjects/perceptron/sin_example/src/sin_example.cpp > CMakeFiles/example.dir/sin_example.cpp.i

sin_example/src/CMakeFiles/example.dir/sin_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/example.dir/sin_example.cpp.s"
	cd /home/davidl09/CLionProjects/perceptron/build/sin_example/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/davidl09/CLionProjects/perceptron/sin_example/src/sin_example.cpp -o CMakeFiles/example.dir/sin_example.cpp.s

# Object files for target example
example_OBJECTS = \
"CMakeFiles/example.dir/sin_example.cpp.o"

# External object files for target example
example_EXTERNAL_OBJECTS =

/home/davidl09/CLionProjects/perceptron/sin_example/example: sin_example/src/CMakeFiles/example.dir/sin_example.cpp.o
/home/davidl09/CLionProjects/perceptron/sin_example/example: sin_example/src/CMakeFiles/example.dir/build.make
/home/davidl09/CLionProjects/perceptron/sin_example/example: sin_example/src/CMakeFiles/example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/davidl09/CLionProjects/perceptron/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/davidl09/CLionProjects/perceptron/sin_example/example"
	cd /home/davidl09/CLionProjects/perceptron/build/sin_example/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sin_example/src/CMakeFiles/example.dir/build: /home/davidl09/CLionProjects/perceptron/sin_example/example
.PHONY : sin_example/src/CMakeFiles/example.dir/build

sin_example/src/CMakeFiles/example.dir/clean:
	cd /home/davidl09/CLionProjects/perceptron/build/sin_example/src && $(CMAKE_COMMAND) -P CMakeFiles/example.dir/cmake_clean.cmake
.PHONY : sin_example/src/CMakeFiles/example.dir/clean

sin_example/src/CMakeFiles/example.dir/depend:
	cd /home/davidl09/CLionProjects/perceptron/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/davidl09/CLionProjects/perceptron /home/davidl09/CLionProjects/perceptron/sin_example/src /home/davidl09/CLionProjects/perceptron/build /home/davidl09/CLionProjects/perceptron/build/sin_example/src /home/davidl09/CLionProjects/perceptron/build/sin_example/src/CMakeFiles/example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : sin_example/src/CMakeFiles/example.dir/depend

