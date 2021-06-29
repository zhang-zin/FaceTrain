cmake_minimum_required(VERSION 3.17)
project(FaceTrain)

set(CMAKE_CXX_STANDARD 14)

add_executable(FaceTrain main.cpp)

set(OpenCV_DIR E:/opencv/mingw-build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(FaceTrain ${OpenCV_LIBS})