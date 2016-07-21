#!/bin/sh

g++ test.cpp -o test.out -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -I /usr/local/include/ -L /usr/local/lib && ./test.out
