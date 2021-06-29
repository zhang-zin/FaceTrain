cmake_minimum_required(VERSION 3.15)
project(FaceTrain)

set(CMAKE_CXX_STANDARD 11)

add_executable(FaceTrain main.cpp)
#修改为自己的路径
set(OpenCV_DIR F:/Lance/OpenCV/mingw-build2)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")


target_link_libraries( FaceTrain ${OpenCV_LIBS} )