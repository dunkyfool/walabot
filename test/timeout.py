import sys, select

print "You have five seconds to answer!"

i, o, e = select.select( [sys.stdin], [], [], 5 )

if (i):
  a = sys.stdin.readline().strip()
  #print "You said", sys.stdin.readline().strip()
else:
  print "You said nothing!"

print a
print type(a)

if a=='y': print 'fuck'
