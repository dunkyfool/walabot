import multiprocessing as mp
import psutil as ps
import os
import time

def log(e,e2,prev_cpu,prev_mem,prev_swap,ctr,state):
  while True:
    if state.value==0 and e.is_set():
      state.value = 1
    elif state.value==0 and not e.is_set():
      pass
    elif state.value==1 and e.is_set():
      prev_cpu.value = (prev_cpu.value*ctr.value +  ps.cpu_percent()) / (ctr.value+1)
      prev_mem.value = (prev_mem.value*ctr.value +  ps.virtual_memory().percent) / (ctr.value+1)
      prev_swap.value = (prev_swap.value*ctr.value +  ps.swap_memory().percent) / (ctr.value+1)
      ctr.value += 1
    elif state.value==1 and not e.is_set():
      state.value = 0
      cmd1 = 'echo cpu:'+str(prev_cpu.value)+'>> tmp'
      cmd2 = 'echo mem:'+str(prev_mem.value)+'>> tmp'
      cmd3 = 'echo swap:'+str(prev_swap.value)+'>> tmp'
      os.system(cmd1)
      os.system(cmd2)
      os.system(cmd3)
#      print prev_cpu.value
#      print prev_mem.value
#      print prev_swap.value
#      print ctr.value
      prev_cpu.value=0
      prev_mem.value=0
      prev_swap.value=0
      ctr.value=0

    if e2.is_set():
      break

def ctrl(e,flag):
  if flag:
    e.set()
  else:
    e.clear()


if __name__ == '__main__':
    pass
    '''
    e = mp.Event()
    e2 = mp.Event()
    prev_cpu = mp.Value('d',0.0)
    prev_mem = mp.Value('d',0.0)
    prev_swap = mp.Value('d',0.0)
    ctr = mp.Value('d',0.0)
    state = mp.Value('d',0.0)

    p1 = mp.Process(target=log, args=(e,e2,prev_cpu,prev_mem,prev_swap,ctr,state))
    p1.start()

    for i in range(10):
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
    '''
