#!bin/sh
while true
do
  cat tmp | grep start |wc -l && cat tmp | grep cpu|wc -l
  tail -n4 tmp| head -n3
  sleep 1
  clear
done
