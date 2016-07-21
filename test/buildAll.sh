#/bin/sh

g++ -o sensorTarget.out SensorCodeSample.cpp -O2 -D__LINUX__ -lWalabotAPI -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

#g++ -o sensorBreathing.out SensorBreathingSampleCode.cpp -O2 -D__LINUX__ -lWalabotAPI

#g++ -o inWall.out InWallSampleCode.cpp -O2 -D__LINUX__ -lWalabotAPI

