FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda-arm64:10.2-cudnn8-devel-ubuntu18.04
# FROM nvcr.io/nvidia/l4t-base:r32.5.0

ENV DEBIAN_FRONTEND=noninteractive

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y --no-install-recommends \
    libeigen3-dev \
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
    libssl-dev \
    xorg-dev libglu1-mesa-dev \
    libusb-1.0-0-dev \
    wget vim \
    libgl1-mesa-dev libglew-dev \
    git \
    lsof \
    software-properties-common \
    valgrind && \
    rm -rf /var/lib/apt/lists/*

# Installing CMAKE library

RUN apt remove --purge --auto-remove cmake && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

# RUN apt remove --purge cmake

# RUN cd $HOME && uname -m && \
#     wget -q https://github.com/Kitware/CMake/releases/download/v3.19.7/cmake-3.19.7.tar.gz && \
#     tar -xzf cmake-3.19.7.tar.gz && ls -alh

# RUN cd $HOME && cd cmake-3.19.7 && ls -alh &&\
#     chmod +x bootstrap && ls -alh && \
#     ./bootstrap --parallel=$(nproc) && \
#     make -j$(nproc) && \
#     make install


# Installing Realsense library

RUN git clone https://github.com/IntelRealSense/librealsense.git && \
    echo `pwd` && ls &&\
    cd librealsense && \
    git checkout v2.42.0 && \
    mkdir build && \
    cd build && \
    cmake ../ \
    -D FORCE_RSUSB_BACKEND=true \
    -D CMAKE_BUILD_TYPE=release \
    -D BUILD_EXAMPLES=true \
    -D OpenGL_GL_PREFERENCE=GLVND \
    -D CMAKE_CUDA_ARCHITECTURES=52 \
    -D BUILD_WITH_CUDA=true && \
    make -j$(nproc) && \
    make install && \
    cd ../../ && rm -rf librealsense


# Installi
# installing Cuda samples
RUN cd $HOME && wget -q https://github.com/NVIDIA/cuda-samples/archive/v10.2.tar.gz && \
    tar -xzvf v10.2.tar.gz && \
    mv cuda-samples-10.2 cuda-samples && \
    rm -rf v10.2.tar.gz

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Install opencv
# RUN apt-get update && \
#     apt-get install -y libopencv-dev \
#     libopencv-core-dev \
#     libopencv-calib3d-dev \
#     libopencv-features2d-dev \
#     libopencv-imgcodecs-dev \
#     libopencv-imgproc-dev && \
#     rm -rf /var/lib/apt/lists/*

