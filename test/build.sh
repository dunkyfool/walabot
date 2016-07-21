#!/bin/sh

#g++ -stdlib=libstdc++ o.cpp -o video.out -I/usr/local/include -l opencv_core -l opencv_imgproc -l opencv_highgui -l opencv_imgcodecs -l opencv_video -lstdc++ -L/usr/local/lib/ -v

g++ -ggdb `pkg-config --cflags --libs opencv3` test.cpp -o video.out && ./video.out
