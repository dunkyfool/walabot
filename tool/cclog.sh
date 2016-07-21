#!/bin/sh

#i=0
#while [ $i -lt 40 ]
#do
#  head -n$i walabot.log| tail -n1 walabot.log|grep -o ' '|wc -l
#  i=$((i+1))
#done

cat walabot.log|wc -l
