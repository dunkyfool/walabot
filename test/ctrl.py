import multiprocessing
import os
import psutil

def worker(file):
  cmd = 'python '+file+' &'
  os.system(cmd)

if __name__ == '__main__':
  if not os.path.isdir('pid/'):
    cmd = 'mkdir pid'
    os.system(cmd)

  files = ['tool/run.py','tool/mem.py']
  _pid = ['pid/runpid','pid/mempid']
  pid = []

  for i in files:
    p1 = multiprocessing.Process(target=worker(i))
    p1.start()

  os.system('sleep 1')

  for i in _pid:
    f = open(i,'r')
    pid += [int(f.read())]
    f.close()

  print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
  print pid[0], pid[1]
  print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
  while True:
    if not psutil.pid_exists(pid[0]):
      cmd = 'kill -9 ' + str(pid[1])
      os.system(cmd)
      print 'program over'
      break
