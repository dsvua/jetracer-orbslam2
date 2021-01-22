
all: program

program: add_cuda.o
	g++ -o add -L/usr/local/cuda/lib64 -lcuda -lcudart main.cpp add_cuda.o

add_cuda.o:
	nvcc -c add_cuda.cu

clean: 
	rm -f *.o add

grayscale: cudaYUVtoGray.o
# 	g++ -lstdc++ -std=c++17 -o grayscale -lcuda grayscale.cpp -I../include/cuda -I/usr/local/cuda/include/ -I/usr/include/ -L/usr/local/cuda/lib -L/usr/lib/x86_64-linux-gnu
	g++ -lstdc++ -std=c++11 -I/usr/include/ -I../include/cuda -I/usr/local/cuda/include/ -L/usr/local/lib/ -L/usr/local/cuda/lib64 -g -o grayscale grayscale.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_imgcodecs -lcuda -lcudart

cudaYUVtoGray.o:
	nvcc -c ../src/cuda/cudaYUVtoGray.cu -I../include/cuda -L/usr/local/cuda/lib64


# g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -I../include/cuda -L/usr/local/lib/ -L/usr/local/cuda/lib -g -o grayscale  grayscale.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching