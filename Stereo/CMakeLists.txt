#Project
cmake_minimum_required(VERSION 2.8)
project( Stereo )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O3") 
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

#Set OpenCV_DIR to the folder of OpenCVConfig.cmake.
#Therefore, find_package will automatically read the cmake file for configuration.
#Several varaibles will be set up. More detail : OpenCVConfig.cmake
#Those included files have been done by OpenCVConfig.cmake as well.

#----------------------------------------------------------------------------------

SET(OpenCV_DIR "/home/sanyu/opencv/opencv-3.1.0/share/OpenCV") #local level

find_package( OpenCV REQUIRED )

message("-- OpenCVConfig.cmake :${OpenCV_CONFIG}")
message("-- OpenCV_INCLUDE_DIRS :${OpenCV_INCLUDE_DIRS}")
message("-- OpenCV_INSTALL_PATH :${OpenCV_INSTALL_PATH}")
message("-- OpenCV_LIBS_PATH :${OpenCV_LIBS_PATH}")

add_executable( sgm_stereo sgm_stereo.cpp SGM.cpp )
target_link_libraries( sgm_stereo ${OpenCV_LIBS})

