import psutil as ps
import os
import numpy as np

pid=os.getpid()
f = open('pid/mempid','w')
f.write(str(pid))
f.close()

while True:
  os.system('sleep 0.01')

  print os.getpid()
  p = ps.Process(pid)
  #print p.memory_full_info()
  #print p.threads()
  a = np.zeros((1000,1000))
  b = np.zeros((1000,1000))
  c = a + b



