FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    glmark2 \
    libasio-dev \
    libwebsocketpp-dev \
    build-essential \
    gdb qt5dxcb-plugin libxml2 \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    xorg-dev libglu1-mesa-dev \
    libusb-1.0-0-dev \
    wget vim \
    libgl1-mesa-dev libglew-dev \
    git \
    cuda-nsight-systems-10-2 cuda-nsight-compute-10-2 cuda-visual-tools-10-2 && \
    rm -rf /var/lib/apt/lists/*

# Installing CMAKE library

RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh && \
    chmod +x cmake-3.18.2-Linux-x86_64.sh; ./cmake-3.18.2-Linux-x86_64.sh --skip-license && \
    rm -rf cmake-3.18.2-Linux-x86_64.sh

# Installing Realsense library

RUN git clone https://github.com/IntelRealSense/librealsense.git && \
    echo `pwd` && ls &&\
    cd librealsense && ls &&\
    mkdir build &&\
    cd build && \
    cmake ../ -D FORCE_RSUSB_BACKEND=true \
    -D CMAKE_BUILD_TYPE=release \
    -D BUILD_EXAMPLES=true \
    -D OpenGL_GL_PREFERENCE=GLVND \
    -D CMAKE_CUDA_ARCHITECTURES=52 \
    -D BUILD_WITH_CUDA=true && \
    make -j$(nproc) && \
    make install && \
    cd ../../ && rm -rf librealsense cmake-3.18.2-Linux-x86_64.sh


# Installing OpenCV

ENV OPENCV_VERSION=4.5.1
ENV ARCH_BIN=5.2
ENV INSTALL_DIR=/usr/local
ENV DOWNLOAD_OPENCV_EXTRAS=YES
ENV OPENCV_SOURCE_DIR=$HOME
ENV WHEREAMI=$PWD
ENV CLEANUP=true
ENV PACKAGE_OPENCV="-D CPACK_BINARY_DEB=ON"
ENV CMAKE_INSTALL_PREFIX=$INSTALL_DIR
# ENV 
# ENV 
# ENV 
# ENV 

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libeigen3-dev \
    libglew-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libpostproc-dev \
    libswscale-dev \
    libtbb-dev \
    libtiff5-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libwebp-dev \
    zlib1g-dev \
    pkg-config &&\
    rm -rf /var/lib/apt/lists/*

RUN cd $HOME && \
    git clone --branch "4.5.1" https://github.com/opencv/opencv.git && \
    git clone --branch "4.5.1" https://github.com/opencv/opencv_contrib.git && \
    ls && ls opencv_contrib && \
    cd $HOME/opencv && \
    sed -i 's/include <Eigen\/Core>/include <eigen3\/Eigen\/Core>/g' modules/core/include/opencv2/core/private.hpp && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=5.2 \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_V4L=ON \
    -D WITH_GSTREAMER=OFF \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=YES \
    -D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
    ../ && \
    make -j$(nproc) && \
    make install && \
    ldconfig
# make package -j$NUM_JOBS && \



#RUN cd librealsense; cp config/99-realsense-libusb.rules /etc/udev/rules.d/

# RUN cd librealsense; mkdir build

# RUN cd librealsense/build; cmake ../ -DFORCE_LIBUVC=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_WITH_CUDA=true

# RUN cd librealsense/build; make -j`grep -c ^processor /proc/cpuinfo` && make install

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
