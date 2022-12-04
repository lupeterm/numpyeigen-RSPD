cmake_minimum_required(VERSION 3.15)

# #################################################################
# TODO: You should change the project name to match your project #
# #################################################################
project(numpyeigen-example-project)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Download NumpyEigen if we haven't already
# set(EXTERNAL_DEP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
# include(DownloadExternalDeps)
# download_dep(numpyeigen
# GIT_REPOSITORY https://github.com/fwilliams/numpyeigen.git
# GIT_TAG 4916d926aa2b939bd8f625c7537563a1575dafe9
# )
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/numpyeigen/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(numpyeigen) # This will define Eigen3::Eigen if we enabled NPE_WITH_EIGEN

# ##################################################################################################################
# TODO: You need to change the following code for your module (e.g. to add more sources, or to rename the module) #
# ##################################################################################################################

# Create a Python module called _example_internal which will live in the example/ directory
# You can import functions from the internal module in example/__init__.py to publically expose them
npe_add_module(_example_internal
    BINDING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/example.cpp
)

add_library(rspdlib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/angleutils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/boundaryvolumehierarchy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/circle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/collisiondetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/connection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/connectivitygraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/planedetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/primitive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/cylinder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/cylinderfunctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/extremity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/functor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/geometryutils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/line.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/nearestneighborcalculator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/normalestimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/partitioner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/pcacalculator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/planarpatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/point.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/pointcloud.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/primitivedetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/rect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/statisticsutils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSPD/unionfind.cpp

)

target_include_directories(_example_internal PRIVATE "RSPD")
target_link_libraries(_example_internal PRIVATE rspdlib)

set_target_properties(_example_internal PROPERTIES COMPILE_FLAGS "-fvisibility=hidden -msse3")

enable_testing()
add_subdirectory(tests)
