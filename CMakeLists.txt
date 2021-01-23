cmake_minimum_required(VERSION 2.8.3)
project(jetracer)

#ds determine build type, default build type: release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()
message("${PROJECT_NAME}|build type: '${CMAKE_BUILD_TYPE}'")

#ds flags for release build
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -std=c++11 -Wall -Werror -pedantic -DNDEBUG")
  message("${PROJECT_NAME}|adding flags for '${CMAKE_BUILD_TYPE}': '-O3 -fPIC -std=c++11 -Wall -Werror -pedantic -DNDEBUG'")

#ds flags for other build(s) (e.g. debug)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -pedantic -O0 -g -fstack-check")
  message("${PROJECT_NAME}|adding flags for '${CMAKE_BUILD_TYPE}': '-std=c++11 -Wall -Werror -pedantic -O0 -g -fstack-check'")
endif()

#ds enable ARM flags if applicable
if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -funsafe-math-optimizations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -funsafe-math-optimizations")
  message("${PROJECT_NAME}|enabling ARM neon optimizations")
endif()

#ds load Eigen library
#find_package(Eigen3 REQUIRED)
#message("${PROJECT_NAME}|using Eigen version: '3' (${EIGEN3_INCLUDE_DIR})")

#ds set sources
message("${PROJECT_NAME}|final flags: '${CMAKE_CXX_FLAGS}'")
add_subdirectory(src)
