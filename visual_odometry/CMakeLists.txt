cmake_minimum_required( VERSION 2.8)
project( visual_odometry)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# add cmake module to use g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package( OpenCV REQUIRED )
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories(
        ${OpenCV_INCLUDED_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
)

add_executable( feature_extraction_chi feature_extraction_chi.cpp)
target_link_libraries( feature_extraction_chi ${OpenCV_LIBS} )

add_executable( pose_estimation_2d2d_chi pose_estimation_2d2d_chi.cpp)
target_link_libraries( pose_estimation_2d2d_chi ${OpenCV_LIBS} )

add_executable( triangulation_chi triangulation_chi.cpp)
target_link_libraries( triangulation_chi ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d_chi pose_estimation_3d2d_chi.cpp)
target_link_libraries( pose_estimation_3d2d_chi
        ${OpenCV_LIBS}
        ${CSPARSE_LIBRARY}
        g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
        )

add_executable( pose_estimation_3d3d_chi pose_estimation_3d3d_chi.cpp)
target_link_libraries( pose_estimation_3d3d_chi
        ${OpenCV_LIBS}
        ${CSPARSE_LIBRARY}
        g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
        )