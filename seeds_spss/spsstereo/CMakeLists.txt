#Project
cmake_minimum_required(VERSION 2.4)
project(spsstereo)

set(OpenCV_DIR "/home/sanyu/opencv/opencv-2.4.13/share/OpenCV") #local level

find_package( OpenCV REQUIRED )

# Flags
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2")
if (APPLE)
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libstdc++")
endif (APPLE)
set(CMAKE_BUILD_TYPE Release)

file(GLOB PNG_LIBS "/usr/local/lib/libpng16.so")
set (PNG_LINK_LIBS ${PNG_LIBS})


# SPS-Stereo
#file(GLOB SPSSTEREO_SRC_FILES "*.cpp")
add_executable(spsstereo spsstereo_main.cpp SPSStereo.cpp SGMStereo.cpp)
target_link_libraries(spsstereo ${PNG_LINK_LIBS} ${OpenCV_LIBS})



