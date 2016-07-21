import os

pid=os.getpid()
f = open('pid/runpid','w')
f.write(str(pid))
f.close()

ctr=0
while ctr < 100:
  if ctr%3==0:
    print 'run.'
  elif ctr%3==1:
    print 'run..'
  elif ctr%3==2:
    print 'run...'
  ctr += 1
  os.system('sleep 0.01')
