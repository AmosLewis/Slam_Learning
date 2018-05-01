cmake_minimum_required(VERSION 3.9)
project(ceres_curve_fitting_chi)

set(CMAKE_CXX_STANDARD 11)
# add cmake_modules to use ceres
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# find ceres and add ceres include
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable(ceres_curve_fitting_chi main.cpp)
target_link_libraries( ceres_curve_fitting_chi ${CERES_LIBRARIES} ${OpenCV_LIBS} )