all: program

program: 
	g++ -std=c++11 src/calibrate.cpp -o build/calibrate -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d


clean: 
	rm -f build/*
