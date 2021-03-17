cmake_minimum_required(VERSION 3.2)
PROJECT(sfm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

# opencv
FIND_PACKAGE(OpenCV 4 REQUIRED)
if (OPENCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_definitions(-DENABLE_OPENCV)
    list(APPEND LINK_LIB_LIST ${OpenCV_LIBS})
endif (OPENCV_FOUND)


find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


include_directories(
    ./include
    ./
    ./include/eigen3/
    )

AUX_SOURCE_DIRECTORY(src DIR_SRCS)
SET(SRC_DIR ${DIR_SRCS})

link_directories(lib)
link_libraries()

SET(EXTRA_LIBS 
    ${EXTRA_LIBS} 
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    )

file(GLOB example_files RELATIVE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/app/*.cpp
    )
	
foreach(example_file ${example_files})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file} ${SRC_DIR})
    target_link_libraries(${example_name} ${EXTRA_LIBS})
endforeach(example_file ${example_files})