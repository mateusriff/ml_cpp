cmake_minimum_required(VERSION 3.10)
project(ml_cpp)

set(SRC_DIR src)
set(INCLUDE_DIR include/ml_cpp)

file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.cpp)

add_library(ml_lib STATIC ${SOURCES})
target_include_directories(ml_lib PUBLIC ${INCLUDE_DIR})