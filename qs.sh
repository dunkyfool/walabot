#!/bin/sh

i="0"
while [ $i -lt 20 ]
do
  echo $i
  python main.py
  #echo $i
  i=$(($i+1))
done
