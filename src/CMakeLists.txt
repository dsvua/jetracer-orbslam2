
ADD_LIBRARY(${PROJECT_NAME}_library
    Events/EventTypes.cpp
    Fault.cpp
    EventsThread.cpp
    MainEventsLoop.cpp
    PingPong/PingPong.cpp
    RealSense/RealSenseD400.cpp
    WebSocket/WebSocketCom.cpp
    WebSocket/bson.cpp
)

ADD_LIBRARY(${PROJECT_NAME}_library_gpu
    # cuda/jetracer_rscuda_utils.cu
    cuda/orb.cu
    cuda/cuda-align.cu
    cuda/cuda_RGB_to_Grayscale.cu
    cuda/gaussian_blur_3x3.cu
    cuda/pyramid.cu
    cuda/fast.cu
    cuda/nms.cu
    cuda/post_processing.cu
    RealSense/SaveRawData.cpp
    SlamGpuPipeline/SlamGpuPipeline.cpp
    SlamGpuPipeline/buildStream.cpp
    cuda_common.cpp
)

#ds main executable (pthread is used for the GUI thread)
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)

# TARGET_COMPILE_FEATURES(${PROJECT_NAME} PUBLIC cxx_std_17)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_library ${PROJECT_NAME}_library_gpu ${realsense2_LIBRARY} -pthread)
# TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_library ${PROJECT_NAME}_library_gpu Eigen3::Eigen ${realsense2_LIBRARY} ${OpenCV_LIBS} ${CUDART_LIBRARY} ${CUDADEVRT_LIBRARY} -pthread)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_library PUBLIC ${realsense2_LIBRARY} -pthread)
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# TARGET_LINK_LIBRARIES(${PROJECT_NAME}_library_gpu PUBLIC jetracer_library ${PROJECT_NAME}_vilib Eigen3::Eigen ${realsense2_LIBRARY} ${OpenCV_LIBS} ${CUDART_LIBRARY} ${CUDADEVRT_LIBRARY} -pthread)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}_library_gpu PUBLIC jetracer_library ${realsense2_LIBRARY} nvjpeg -pthread)
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES CUDA_ARCHITECTURES 52)
SET_SOURCE_FILES_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES LANGUAGE CUDA)

message(STATUS "CUDA_cudadevrt_LIBRARY ${CUDA_cudadevrt_LIBRARY}")
message(STATUS "CUDADEVRT_LIBRARY ${CUDADEVRT_LIBRARY}")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/external" "/root/cuda-samples/Common" Eigen3::Eigen)
