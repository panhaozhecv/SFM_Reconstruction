# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/panhaozhe/code/C_MAKE/sfmReconstruction

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/panhaozhe/code/C_MAKE/sfmReconstruction/build

# Include any dependencies generated for this target.
include CMakeFiles/run_sfm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/run_sfm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run_sfm.dir/flags.make

CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o: CMakeFiles/run_sfm.dir/flags.make
CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o: ../app/run_sfm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o -c /home/panhaozhe/code/C_MAKE/sfmReconstruction/app/run_sfm.cpp

CMakeFiles/run_sfm.dir/app/run_sfm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_sfm.dir/app/run_sfm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/panhaozhe/code/C_MAKE/sfmReconstruction/app/run_sfm.cpp > CMakeFiles/run_sfm.dir/app/run_sfm.cpp.i

CMakeFiles/run_sfm.dir/app/run_sfm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_sfm.dir/app/run_sfm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/panhaozhe/code/C_MAKE/sfmReconstruction/app/run_sfm.cpp -o CMakeFiles/run_sfm.dir/app/run_sfm.cpp.s

CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.requires:

.PHONY : CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.requires

CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.provides: CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_sfm.dir/build.make CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.provides.build
.PHONY : CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.provides

CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.provides.build: CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o


CMakeFiles/run_sfm.dir/src/baEngine.cpp.o: CMakeFiles/run_sfm.dir/flags.make
CMakeFiles/run_sfm.dir/src/baEngine.cpp.o: ../src/baEngine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/run_sfm.dir/src/baEngine.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_sfm.dir/src/baEngine.cpp.o -c /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/baEngine.cpp

CMakeFiles/run_sfm.dir/src/baEngine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_sfm.dir/src/baEngine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/baEngine.cpp > CMakeFiles/run_sfm.dir/src/baEngine.cpp.i

CMakeFiles/run_sfm.dir/src/baEngine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_sfm.dir/src/baEngine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/baEngine.cpp -o CMakeFiles/run_sfm.dir/src/baEngine.cpp.s

CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.requires:

.PHONY : CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.requires

CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.provides: CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_sfm.dir/build.make CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.provides.build
.PHONY : CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.provides

CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.provides.build: CMakeFiles/run_sfm.dir/src/baEngine.cpp.o


CMakeFiles/run_sfm.dir/src/camera.cpp.o: CMakeFiles/run_sfm.dir/flags.make
CMakeFiles/run_sfm.dir/src/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/run_sfm.dir/src/camera.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_sfm.dir/src/camera.cpp.o -c /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/camera.cpp

CMakeFiles/run_sfm.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_sfm.dir/src/camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/camera.cpp > CMakeFiles/run_sfm.dir/src/camera.cpp.i

CMakeFiles/run_sfm.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_sfm.dir/src/camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/camera.cpp -o CMakeFiles/run_sfm.dir/src/camera.cpp.s

CMakeFiles/run_sfm.dir/src/camera.cpp.o.requires:

.PHONY : CMakeFiles/run_sfm.dir/src/camera.cpp.o.requires

CMakeFiles/run_sfm.dir/src/camera.cpp.o.provides: CMakeFiles/run_sfm.dir/src/camera.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_sfm.dir/build.make CMakeFiles/run_sfm.dir/src/camera.cpp.o.provides.build
.PHONY : CMakeFiles/run_sfm.dir/src/camera.cpp.o.provides

CMakeFiles/run_sfm.dir/src/camera.cpp.o.provides.build: CMakeFiles/run_sfm.dir/src/camera.cpp.o


CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o: CMakeFiles/run_sfm.dir/flags.make
CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o: ../src/matchEngine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o -c /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/matchEngine.cpp

CMakeFiles/run_sfm.dir/src/matchEngine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_sfm.dir/src/matchEngine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/matchEngine.cpp > CMakeFiles/run_sfm.dir/src/matchEngine.cpp.i

CMakeFiles/run_sfm.dir/src/matchEngine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_sfm.dir/src/matchEngine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/matchEngine.cpp -o CMakeFiles/run_sfm.dir/src/matchEngine.cpp.s

CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.requires:

.PHONY : CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.requires

CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.provides: CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_sfm.dir/build.make CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.provides.build
.PHONY : CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.provides

CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.provides.build: CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o


CMakeFiles/run_sfm.dir/src/mulThread.cpp.o: CMakeFiles/run_sfm.dir/flags.make
CMakeFiles/run_sfm.dir/src/mulThread.cpp.o: ../src/mulThread.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/run_sfm.dir/src/mulThread.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_sfm.dir/src/mulThread.cpp.o -c /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/mulThread.cpp

CMakeFiles/run_sfm.dir/src/mulThread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_sfm.dir/src/mulThread.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/mulThread.cpp > CMakeFiles/run_sfm.dir/src/mulThread.cpp.i

CMakeFiles/run_sfm.dir/src/mulThread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_sfm.dir/src/mulThread.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/panhaozhe/code/C_MAKE/sfmReconstruction/src/mulThread.cpp -o CMakeFiles/run_sfm.dir/src/mulThread.cpp.s

CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.requires:

.PHONY : CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.requires

CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.provides: CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_sfm.dir/build.make CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.provides.build
.PHONY : CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.provides

CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.provides.build: CMakeFiles/run_sfm.dir/src/mulThread.cpp.o


# Object files for target run_sfm
run_sfm_OBJECTS = \
"CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o" \
"CMakeFiles/run_sfm.dir/src/baEngine.cpp.o" \
"CMakeFiles/run_sfm.dir/src/camera.cpp.o" \
"CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o" \
"CMakeFiles/run_sfm.dir/src/mulThread.cpp.o"

# External object files for target run_sfm
run_sfm_EXTERNAL_OBJECTS =

run_sfm: CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o
run_sfm: CMakeFiles/run_sfm.dir/src/baEngine.cpp.o
run_sfm: CMakeFiles/run_sfm.dir/src/camera.cpp.o
run_sfm: CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o
run_sfm: CMakeFiles/run_sfm.dir/src/mulThread.cpp.o
run_sfm: CMakeFiles/run_sfm.dir/build.make
run_sfm: /usr/local/lib/libopencv_gapi.so.4.1.2
run_sfm: /usr/local/lib/libopencv_stitching.so.4.1.2
run_sfm: /usr/local/lib/libopencv_aruco.so.4.1.2
run_sfm: /usr/local/lib/libopencv_bgsegm.so.4.1.2
run_sfm: /usr/local/lib/libopencv_bioinspired.so.4.1.2
run_sfm: /usr/local/lib/libopencv_ccalib.so.4.1.2
run_sfm: /usr/local/lib/libopencv_dnn_objdetect.so.4.1.2
run_sfm: /usr/local/lib/libopencv_dnn_superres.so.4.1.2
run_sfm: /usr/local/lib/libopencv_dpm.so.4.1.2
run_sfm: /usr/local/lib/libopencv_face.so.4.1.2
run_sfm: /usr/local/lib/libopencv_freetype.so.4.1.2
run_sfm: /usr/local/lib/libopencv_fuzzy.so.4.1.2
run_sfm: /usr/local/lib/libopencv_hfs.so.4.1.2
run_sfm: /usr/local/lib/libopencv_img_hash.so.4.1.2
run_sfm: /usr/local/lib/libopencv_line_descriptor.so.4.1.2
run_sfm: /usr/local/lib/libopencv_quality.so.4.1.2
run_sfm: /usr/local/lib/libopencv_reg.so.4.1.2
run_sfm: /usr/local/lib/libopencv_rgbd.so.4.1.2
run_sfm: /usr/local/lib/libopencv_saliency.so.4.1.2
run_sfm: /usr/local/lib/libopencv_stereo.so.4.1.2
run_sfm: /usr/local/lib/libopencv_structured_light.so.4.1.2
run_sfm: /usr/local/lib/libopencv_superres.so.4.1.2
run_sfm: /usr/local/lib/libopencv_surface_matching.so.4.1.2
run_sfm: /usr/local/lib/libopencv_tracking.so.4.1.2
run_sfm: /usr/local/lib/libopencv_videostab.so.4.1.2
run_sfm: /usr/local/lib/libopencv_xfeatures2d.so.4.1.2
run_sfm: /usr/local/lib/libopencv_xobjdetect.so.4.1.2
run_sfm: /usr/local/lib/libopencv_xphoto.so.4.1.2
run_sfm: /usr/local/lib/libceres.a
run_sfm: /usr/local/lib/libopencv_shape.so.4.1.2
run_sfm: /usr/local/lib/libopencv_highgui.so.4.1.2
run_sfm: /usr/local/lib/libopencv_datasets.so.4.1.2
run_sfm: /usr/local/lib/libopencv_plot.so.4.1.2
run_sfm: /usr/local/lib/libopencv_text.so.4.1.2
run_sfm: /usr/local/lib/libopencv_dnn.so.4.1.2
run_sfm: /usr/local/lib/libopencv_ml.so.4.1.2
run_sfm: /usr/local/lib/libopencv_phase_unwrapping.so.4.1.2
run_sfm: /usr/local/lib/libopencv_optflow.so.4.1.2
run_sfm: /usr/local/lib/libopencv_ximgproc.so.4.1.2
run_sfm: /usr/local/lib/libopencv_video.so.4.1.2
run_sfm: /usr/local/lib/libopencv_videoio.so.4.1.2
run_sfm: /usr/local/lib/libopencv_imgcodecs.so.4.1.2
run_sfm: /usr/local/lib/libopencv_objdetect.so.4.1.2
run_sfm: /usr/local/lib/libopencv_calib3d.so.4.1.2
run_sfm: /usr/local/lib/libopencv_features2d.so.4.1.2
run_sfm: /usr/local/lib/libopencv_flann.so.4.1.2
run_sfm: /usr/local/lib/libopencv_photo.so.4.1.2
run_sfm: /usr/local/lib/libopencv_imgproc.so.4.1.2
run_sfm: /usr/local/lib/libopencv_core.so.4.1.2
run_sfm: /usr/lib/x86_64-linux-gnu/libglog.so
run_sfm: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
run_sfm: /usr/lib/x86_64-linux-gnu/libspqr.so
run_sfm: /usr/lib/x86_64-linux-gnu/libtbb.so
run_sfm: /usr/lib/x86_64-linux-gnu/libcholmod.so
run_sfm: /usr/lib/x86_64-linux-gnu/libccolamd.so
run_sfm: /usr/lib/x86_64-linux-gnu/libcamd.so
run_sfm: /usr/lib/x86_64-linux-gnu/libcolamd.so
run_sfm: /usr/lib/x86_64-linux-gnu/libamd.so
run_sfm: /usr/lib/x86_64-linux-gnu/liblapack.so
run_sfm: /usr/lib/x86_64-linux-gnu/libf77blas.so
run_sfm: /usr/lib/x86_64-linux-gnu/libatlas.so
run_sfm: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
run_sfm: /usr/lib/x86_64-linux-gnu/librt.so
run_sfm: /usr/lib/x86_64-linux-gnu/libcxsparse.so
run_sfm: /usr/lib/x86_64-linux-gnu/liblapack.so
run_sfm: /usr/lib/x86_64-linux-gnu/libf77blas.so
run_sfm: /usr/lib/x86_64-linux-gnu/libatlas.so
run_sfm: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
run_sfm: /usr/lib/x86_64-linux-gnu/librt.so
run_sfm: /usr/lib/x86_64-linux-gnu/libcxsparse.so
run_sfm: CMakeFiles/run_sfm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable run_sfm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run_sfm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run_sfm.dir/build: run_sfm

.PHONY : CMakeFiles/run_sfm.dir/build

CMakeFiles/run_sfm.dir/requires: CMakeFiles/run_sfm.dir/app/run_sfm.cpp.o.requires
CMakeFiles/run_sfm.dir/requires: CMakeFiles/run_sfm.dir/src/baEngine.cpp.o.requires
CMakeFiles/run_sfm.dir/requires: CMakeFiles/run_sfm.dir/src/camera.cpp.o.requires
CMakeFiles/run_sfm.dir/requires: CMakeFiles/run_sfm.dir/src/matchEngine.cpp.o.requires
CMakeFiles/run_sfm.dir/requires: CMakeFiles/run_sfm.dir/src/mulThread.cpp.o.requires

.PHONY : CMakeFiles/run_sfm.dir/requires

CMakeFiles/run_sfm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run_sfm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run_sfm.dir/clean

CMakeFiles/run_sfm.dir/depend:
	cd /home/panhaozhe/code/C_MAKE/sfmReconstruction/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/panhaozhe/code/C_MAKE/sfmReconstruction /home/panhaozhe/code/C_MAKE/sfmReconstruction /home/panhaozhe/code/C_MAKE/sfmReconstruction/build /home/panhaozhe/code/C_MAKE/sfmReconstruction/build /home/panhaozhe/code/C_MAKE/sfmReconstruction/build/CMakeFiles/run_sfm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run_sfm.dir/depend
