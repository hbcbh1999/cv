cmake_minimum_required(VERSION 2.4)

PROJECT(SEEDS)

#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)
#SET(EXECUTABLE_OUTPUT_PATH ~/bin/EM
#    CACHE PATH "Single output directory for building all executables.")

#set(CMAKE_CXX_FLAGS         " -O3")
#set(CMAKE_CXX_FLAGS_DEBUG   "-g -pg ")
#set(CMAKE_CXX_FLAGS_RELEASE " -pg -O3")

#set(CMAKE_C_FLAGS         ${CMAKE_CXX_FLAGS})
#set(CMAKE_C_FLAGS_DEBUG   ${CMAKE_CXX_FLAGS_DEBUG})
#set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2")
SET(OpenCV_DIR "/home/sanyu/opencv/opencv-3.1.0/share/OpenCV")
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_DIR})

file(GLOB PNG_LIBS "/usr/local/lib/libpng16.so")
set (PNG_LINK_LIBS ${PNG_LIBS})


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


ADD_EXECUTABLE(main 
main.cpp
SEED/seeds2.cpp
spsstereo/SPSStereo.cpp
spsstereo/SGMStereo.cpp
)

TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${PNG_LINK_LIBS})


