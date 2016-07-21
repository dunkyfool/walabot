import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def sameObject_diffR(a1,a2,a3,a4,a5,a6,a7,a8,size,idx):
  a1 = a1.mean(axis=0)
  a2 = a2.mean(axis=0)
  a3 = a3.mean(axis=0)
  a4 = a4.mean(axis=0)
  a5 = a5.mean(axis=0)
  a6 = a6.mean(axis=0)
  a7 = a7.mean(axis=0)
  a8 = a8.mean(axis=0)

  m = np.mean((a1,a2,a3,a4,a5,a6,a7,a8),axis=0)
  a1 = abs(a1-m)
  a2 = abs(a2-m)
  a3 = abs(a3-m)
  a4 = abs(a4-m)
  a5 = abs(a5-m)
  a6 = abs(a6-m)
  a7 = abs(a7-m)
  a8 = abs(a8-m)

  plt.clf()
  plt.subplot(811)
  plt.title('{0}30'.format(size))
  plt.plot(np.arange(a1.shape[0]),a1)

  plt.subplot(812)
  plt.title('{0}40'.format(size))
  plt.plot(np.arange(a2.shape[0]),a2)

  plt.subplot(813)
  plt.title('{0}50'.format(size))
  plt.plot(np.arange(a3.shape[0]),a3)

  plt.subplot(814)
  plt.title('{0}60'.format(size))
  plt.plot(np.arange(a4.shape[0]),a4)

  plt.subplot(815)
  plt.title('{0}70'.format(size))
  plt.plot(np.arange(a5.shape[0]),a5)

  plt.subplot(816)
  plt.title('{0}80'.format(size))
  plt.plot(np.arange(a6.shape[0]),a6)

  plt.subplot(817)
  plt.title('{0}90'.format(size))
  plt.plot(np.arange(a7.shape[0]),a7)

  plt.subplot(818)
  plt.title('{0}100'.format(size))
  plt.plot(np.arange(a8.shape[0]),a8)

  plt.savefig('data/sml_1/size_{0}_pair_{1}.png'.format(size,idx))

def diffObject_sameR(a1,a2,a3,size,idx):
  a1 = a1.mean(axis=0)
  a2 = a2.mean(axis=0)
  a3 = a3.mean(axis=0)

  m = np.mean((a1,a2,a3),axis=0)
  a1 = abs(a1-m)
  a2 = abs(a2-m)
  a3 = abs(a3-m)

  plt.clf()

  plt.subplot(311)
  plt.title('s{0}'.format(size))
  plt.plot(np.arange(a1.shape[0]),a1)

  plt.subplot(312)
  plt.title('m{0}'.format(size))
  plt.plot(np.arange(a2.shape[0]),a2)

  plt.subplot(313)
  plt.title('l{0}'.format(size))
  plt.plot(np.arange(a3.shape[0]),a3)

  plt.savefig('data/sml_2/dist_{0}_pair_{1}.png'.format(size,idx))

def toggle(k,a,b,c,d,e,idx):
  plt.clf()
  for i in range(k.shape[1]):
    plt.scatter([i]*k.shape[0], k[:,i])
  k_mean = k.mean(axis=0)
  k_std = k.std(axis=0)
  plt.errorbar(np.arange(k.shape[1]),k_mean,yerr=k_std)
  plt.savefig('data/small/plot{0}_{1}.png'.format(idx,0))

  plt.clf()
  for i in range(a.shape[1]):
    plt.scatter([i]*a.shape[0], a[:,i])
  a_mean = a.mean(axis=0)
  a_std = a.std(axis=0)
  plt.errorbar(np.arange(a.shape[1]),a_mean,yerr=a_std)
  plt.savefig('data/small/plot{0}_{1}.png'.format(idx,1))

  plt.clf()
  for i in range(b.shape[1]):
    plt.scatter([i]*b.shape[0], b[:,i])
  b_mean = b.mean(axis=0)
  b_std = b.std(axis=0)
  plt.errorbar(np.arange(b.shape[1]),b_mean,yerr=b_std)
  plt.savefig('data/small/plot{0}_{1}.png'.format(idx,2))

  plt.clf()
  for i in range(c.shape[1]):
    plt.scatter([i]*c.shape[0], c[:,i])
  c_mean = c.mean(axis=0)
  c_std = c.std(axis=0)
  plt.errorbar(np.arange(c.shape[1]),c_mean,yerr=c_std)
  plt.savefig('data/small/plot{0}_{1}.png'.format(idx,3))

  plt.clf()
  for i in range(d.shape[1]):
    plt.scatter([i]*d.shape[0], d[:,i])
  d_mean = d.mean(axis=0)
  d_std = d.std(axis=0)
  plt.errorbar(np.arange(d.shape[1]),d_mean,yerr=d_std)
  plt.savefig('data/small/plot{0}_{1}.png'.format(idx,4))

  plt.clf()
  for i in range(e.shape[1]):
    plt.scatter([i]*e.shape[0], e[:,i])
  e_mean = e.mean(axis=0)
  e_std = e.std(axis=0)
  plt.errorbar(np.arange(e.shape[1]),e_mean,yerr=e_std)
  plt.savefig('data/small/plot{0}_{1}.png'.format(idx,5))
#  line = []
#  label = []
#  for i in range(k.shape[0]):
#    tmp, = plt.plot(np.arange(k.shape[1]),k[i,:],label='Line{0}'.format(i))
#    line += [tmp]
    #label += ['Line{0}'.format(i)]
#  plt.legend(handles=line, bbox_to_anchor=(1, 0.3),loc=2)

#  plt.title('Raw Signal')
#  plt.xlabel('Time')
#  plt.ylabel('Energy')
#  plt.savefig('data/small/plot{0}.png'.format(idx))
#  plt.show()

def dot(k):  
  plt.plot(np.arange(k.shape[0]),k)
#  for i in range(k.shape[0]):
#    tmp, = plt.plot(np.arange(k.shape[1]),k[i,:],label='Line{0}'.format(i))

  plt.title('Raw Signal')
  plt.xlabel('Time')
  plt.ylabel('Energy')
#  plt.legend(handles=line, bbox_to_anchor=(1, 0.3),loc=2)
  plt.savefig('plot.png')
  plt.show()

def compare(a,b,c,i): # show mean 
  plt.clf()
  a = a.mean(axis=0)
  b = b.mean(axis=0)
  c = c.mean(axis=0)

  plt.subplot(311)
  plt.plot(np.arange(a.shape[0]),a,label='1')

  plt.subplot(312)
  plt.plot(np.arange(b.shape[0]),b,label='2')

  plt.subplot(313)
  plt.plot(np.arange(c.shape[0]),c,label='3')

  plt.savefig('data/wv/plot{0}.png'.format(i))
  plt.show()

def diff(a,b,c,idx): # abs(sample 0 - mean)
  plt.clf()
  a_diff = abs(a - a.mean(axis=0))
  b_diff = abs(b - b.mean(axis=0))
  c_diff = abs(c - c.mean(axis=0))

  plt.subplot(311)
  plt.plot(np.arange(a.shape[1]),a_diff[0],label='1')
  plt.subplot(312)
  plt.plot(np.arange(b.shape[1]),b_diff[0],label='2')
  plt.subplot(313)
  plt.plot(np.arange(c.shape[1]),c_diff[0],label='3')
  plt.savefig('data/abs_x_diff/plot{0}.png'.format(idx))
  plt.show()

def std(a,b,c,idx): # all sample's std
  plt.clf()
  a_std = np.std(a,axis=0)
  b_std = np.std(b,axis=0)
  c_std = np.std(c,axis=0)

  plt.subplot(311)
  plt.plot(np.arange(a.shape[1]),a_std,label='1')
  plt.subplot(312)
  plt.plot(np.arange(b.shape[1]),b_std,label='2')
  plt.subplot(313)
  plt.plot(np.arange(c.shape[1]),c_std,label='3')
  plt.savefig('data/std/plot{0}.png'.format(idx))
  plt.show()

def show_fft(a,b,c,idx):
  plt.clf()
  a_fft = np.fft.fft(np.mean(a,axis=0))
  a_freq = np.fft.fftfreq(a_fft.shape[-1],d=1e-11)
  a_FQ = np.sqrt(a_fft.real**2+a_fft.imag**2)
  b_fft = np.fft.fft(np.mean(b,axis=0))
  b_freq = np.fft.fftfreq(b_fft.shape[-1],d=1e-11)
  b_FQ = np.sqrt(b_fft.real**2+b_fft.imag**2)
  c_fft = np.fft.fft(np.mean(c,axis=0))
  c_freq = np.fft.fftfreq(c_fft.shape[-1],d=1e-11)
  c_FQ = np.sqrt(c_fft.real**2+c_fft.imag**2)

  plt.subplot(311)
  plt.plot(a_freq,a_FQ,label='1')
  plt.subplot(312)
  plt.plot(b_freq,b_FQ,label='2')
  plt.subplot(313)
  plt.plot(c_freq,c_FQ,label='3')
  plt.savefig('data/fft/plot{0}.png'.format(idx))
  plt.show()

def p():
  plt.clf()
  t= np.linspace(0,1,500,endpoint=False)
  #sin = np.cos(0.07*t)+2*np.sin(0.05*t)
  sin = signal.square(2 * np.pi * 5 * t)
  fq = np.fft.fft(sin)
  fq2 = np.fft.fftfreq(sin.shape[-1],d=1)
  FQ = np.sqrt(fq.real**2+fq.imag**2)

  plt.subplot(511)
  plt.plot(np.arange(sin.shape[0]),sin,label='1')
  plt.subplot(512)
  plt.plot(np.arange(fq.shape[0]),fq,label='2')
  plt.subplot(513)
  plt.plot(fq2,fq.real,label='3')
  plt.subplot(514)
  plt.plot(fq2,fq.imag,label='4')
  plt.subplot(515)
  plt.plot(fq2,FQ,label='5')
  plt.savefig('data/fft/___.png')

if __name__=='__main__':
  pass
