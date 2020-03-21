all: program

program: calibrate calibrate_fisheye rectify

calibrate:
	g++ -std=c++11 src/calibrate.cpp -o build/calibrate -Iinclude -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_imgcodecs

calibrate_fisheye:
	g++ -std=c++11 src/calibrate_fisheye.cpp -o build/calibrate_fisheye -Iinclude -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_imgcodecs

rectify:
	g++ -std=c++11 src/undistort_rectify.cpp -o build/rectify -Iinclude -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_imgcodecs -lpopt

clean: 
	rm -f build/*
