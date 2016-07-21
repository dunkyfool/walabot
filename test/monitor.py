import psutil
import os
import subprocess

def checktmp(FILE):
  #bullet-proof
  if not os.path.exists(FILE):
    cmd = 'touch tmp'
    os.system(cmd)

  #check last line start/end flag
  cmd = 'tail -n1 '+FILE
  string = os.popen(cmd).read()[:-1]
  #print string,len(string),len(string.split(' ')[-1])
  string = string.split(' ')[-1]
  if string=='[start]':
    #print 'Enter state 1'
    state = 1
    return state
  elif string=='[end]':
    #print 'Enter state 0'
    state = 0
    return state
  return -1

#sudo
def mem():
  #p = psutil.Process(pid)
  #print p.memory_full_info()
  #print p.memory_percent()
  cpu = psutil.cpu_percent()
  mem =  psutil.virtual_memory().percent
  swap =  psutil.swap_memory().percent
  return cpu, mem, swap

def monitor(FILE):
  state = 0
  base_cpu, base_mem, base_swap = None, None, None
  history_cpu, history_mem, history_swap =[],[],[]
  while True:
    os.system('sleep 1')
    if state == 0:
      flag = checktmp(FILE)
      if flag == -1 or flag == 0:
        #base_cpu, base_mem, base_swap = mem()
        print state,'record base'
      elif flag == 1: #start
        state = 1
        print state,'Enter to state 1'
    elif state == 1:
      flag = checktmp(FILE)
      if flag == -1 or flag == 1: #record
        cpu,memo,swap = mem()
        history_cpu += [cpu]#-base_cpu]
        history_mem += [memo]#-base_mem]
        history_swap += [swap]#-base_swap]
        print state,'record cpu/mem/swap'
      elif flag == 0: #end
        avg_cpu = sum(history_cpu) / float(len(history_cpu))
        max_cpu = max(history_cpu)
        del history_cpu[:]
        avg_mem = sum(history_mem) / float(len(history_mem))
        max_mem = max(history_mem)
        del history_mem[:]
        avg_swap = sum(history_swap) / float(len(history_swap))
        max_swap = max(history_swap)
        del history_swap[:]

        line = 'avg_cpu:\t'+str(avg_cpu)+'\tmax_cpu:\t'+str(max_cpu)
        print line
        insert(3,line,FILE)
        line = 'avg_mem:\t'+str(avg_mem)+'\t\tmax_mem:\t'+str(max_mem)
        print line
        insert(3,line,FILE)
        line = 'avg_swap:\t'+str(avg_swap)+'\t\tmax_swap:\t'+str(max_swap)
        print line
        insert(3,line,FILE)

        state = 0
        print state, 'return to state 0'


def insert(opt,title,FILE):
  with open(FILE,'a') as f:
    if opt==1:#start
      line = title + ' [start]\n'
      f.write(line)
    elif opt==2:#end
      line = title + ' [end]\n'
      f.write(line)
    if opt==3:#
      line = title + '\n'
      f.write(line)

if __name__=='__main__':
  pass
  FILE = 'tmp'
  monitor(FILE)
