CMAKE_MINIMUM_REQUIRED(VERSION 3.18.2 FATAL_ERROR)

SET(PROJECT_NAME jetracer)
PROJECT(${PROJECT_NAME} LANGUAGES CUDA CXX)

INCLUDE(GNUInstallDirs)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    # set(CUDA_ARCHITECTURES 50)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)    
endif()

# set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)    
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
link_directories(${CMAKE_CUDA_INCLUDE_DIRS})

# find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# find_library(CUDADEVRT_LIBRARY cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(ASIO_LIBRARY asio)
find_library(WEBSOCKETPP_LIBRARY websocketpp)
# find_library(NPPIG_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_package(realsense2 2.29.0 REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(OpenCV REQUIRED)

# ---[ Use the default installation path if not set.
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# ---[ Build type
IF(NOT CMAKE_BUILD_TYPE) 
    SET(CMAKE_BUILD_TYPE Develop)
ENDIF(NOT CMAKE_BUILD_TYPE)




#ds set sources
message("${PROJECT_NAME}|final flags: '${CMAKE_CXX_FLAGS}'")
add_subdirectory(src)
