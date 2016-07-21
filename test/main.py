import multiprocessing as mp
import psutil as ps
import os
import time

def test1(q,e,e2,prev,ctr,state):
  while True:
    if state.value==0 and e.is_set():
      state.value = 1
    elif state.value==0 and not e.is_set():
      pass
    elif state.value==1 and e.is_set():
      prev.value = (prev.value*ctr.value +  ps.virtual_memory().percent) / (ctr.value+1)
      ctr.value += 1
    elif state.value==1 and not e.is_set():
      state.value = 0
      cmd1 = 'echo '+str(prev.value)+'>> tmp'
      os.system(cmd1)
      print prev.value
      print ctr.value
      prev.value=0
      ctr.value=0

    if e2.is_set():
      break

def ctrl(e,flag):
  if flag:
    e.set()
  else:
    e.clear()


if __name__ == '__main__':
    q = mp.Queue()
    e = mp.Event()
    e2 = mp.Event()
    prev = mp.Value('d',0.0)
    ctr = mp.Value('d',0.0)
    state = mp.Value('d',0.0)

    p1 = mp.Process(target=test1, args=(q,e,e2,prev,ctr,state))
    p1.start()

    for i in range(20):
      if i%2==0:
        print i
        print 'p1',p1.is_alive()
        print 'e',e.is_set()
        time.sleep(1)
        ctrl(e,True)
        print 'p1',p1.is_alive()
        print 'e',e.is_set()
      else:
        print i
        print 'p1',p1.is_alive()
        print 'e',e.is_set()
        time.sleep(1)
        ctrl(e,False)
        print 'p1',p1.is_alive()
        print 'e',e.is_set()

    e2.set()
