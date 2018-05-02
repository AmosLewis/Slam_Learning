cmake_minimum_required(VERSION 3.9)
project(g2o_curve_fitting_chi)

set(CMAKE_CXX_STANDARD 11)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR})
# find G2O
find_package( G2O REQUIRED )
include_directories(${G2O_INCLUDE_DIRS})
#include_directories( "/usr/include/eigen3" )
include_directories(${EIGEN_INCLUDE_DIRS})
# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpencCV_DIRS})

add_executable(g2o_curve_fitting_chi main.cpp)
# link G20 OpenCV
target_link_libraries( g2o_curve_fitting_chi
        ${OpenCV_LIBS}
        g2o_core g2o_stuff
        )
