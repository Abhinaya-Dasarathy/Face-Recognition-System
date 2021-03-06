# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data/apps/cmake/3.6.1/bin/cmake

# The command to remove a file.
RM = /data/apps/cmake/3.6.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/users/adasarat/demo/parallel-face-recognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/users/adasarat/demo/parallel-face-recognition

# Include any dependencies generated for this target.
include CMakeFiles/ParallelFaceRecog.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ParallelFaceRecog.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ParallelFaceRecog.dir/flags.make

CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o: CMakeFiles/ParallelFaceRecog.dir/flags.make
CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o: mpi_eigenfaces.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/users/adasarat/demo/parallel-face-recognition/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o"
	/data/apps/mpi/openmpi-1.8.3/gcc/4.8.2/bin/mpiCC   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o -c /data/users/adasarat/demo/parallel-face-recognition/mpi_eigenfaces.cpp

CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.i"
	/data/apps/mpi/openmpi-1.8.3/gcc/4.8.2/bin/mpiCC  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/users/adasarat/demo/parallel-face-recognition/mpi_eigenfaces.cpp > CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.i

CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.s"
	/data/apps/mpi/openmpi-1.8.3/gcc/4.8.2/bin/mpiCC  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/users/adasarat/demo/parallel-face-recognition/mpi_eigenfaces.cpp -o CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.s

CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.requires:

.PHONY : CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.requires

CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.provides: CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.requires
	$(MAKE) -f CMakeFiles/ParallelFaceRecog.dir/build.make CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.provides.build
.PHONY : CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.provides

CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.provides.build: CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o


# Object files for target ParallelFaceRecog
ParallelFaceRecog_OBJECTS = \
"CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o"

# External object files for target ParallelFaceRecog
ParallelFaceRecog_EXTERNAL_OBJECTS =

ParallelFaceRecog: CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o
ParallelFaceRecog: CMakeFiles/ParallelFaceRecog.dir/build.make
ParallelFaceRecog: /data/apps/cuda/6.0/lib64/libcudart_static.a
ParallelFaceRecog: /usr/lib64/librt.so
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_videostab.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_ts.a
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_superres.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_stitching.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_contrib.so.2.4.9
ParallelFaceRecog: /data/apps/mpi/openmpi-1.8.3/gcc/4.8.2/lib/libmpi_cxx.so
ParallelFaceRecog: /data/apps/mpi/openmpi-1.8.3/gcc/4.8.2/lib/libmpi.so
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_nonfree.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_ocl.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_gpu.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_photo.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_objdetect.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_legacy.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_video.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_ml.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_calib3d.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_features2d.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_highgui.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_imgproc.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_flann.so.2.4.9
ParallelFaceRecog: /data/apps/opencv/2.4.9/lib/libopencv_core.so.2.4.9
ParallelFaceRecog: CMakeFiles/ParallelFaceRecog.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/users/adasarat/demo/parallel-face-recognition/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ParallelFaceRecog"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ParallelFaceRecog.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ParallelFaceRecog.dir/build: ParallelFaceRecog

.PHONY : CMakeFiles/ParallelFaceRecog.dir/build

CMakeFiles/ParallelFaceRecog.dir/requires: CMakeFiles/ParallelFaceRecog.dir/mpi_eigenfaces.cpp.o.requires

.PHONY : CMakeFiles/ParallelFaceRecog.dir/requires

CMakeFiles/ParallelFaceRecog.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ParallelFaceRecog.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ParallelFaceRecog.dir/clean

CMakeFiles/ParallelFaceRecog.dir/depend:
	cd /data/users/adasarat/demo/parallel-face-recognition && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/users/adasarat/demo/parallel-face-recognition /data/users/adasarat/demo/parallel-face-recognition /data/users/adasarat/demo/parallel-face-recognition /data/users/adasarat/demo/parallel-face-recognition /data/users/adasarat/demo/parallel-face-recognition/CMakeFiles/ParallelFaceRecog.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ParallelFaceRecog.dir/depend

