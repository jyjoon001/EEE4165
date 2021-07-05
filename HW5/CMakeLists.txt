cmake_minimum_required(VERSION 3.0)

set(opencv_DIR "C:/opencv-sources/build")
find_package(opencv)

include_directories(${OpenCV_INCLUDE_DIRS})

get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} C CXX)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES *.cpp)
add_executable(${PROJECT_NAME} ${SOURCES} "utils.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
