import tool.plot as p
from tool.log3 import *
import numpy as np
#'''
trainData,trainLabel,valData,valLabel,testData,testLabel = load()

raw1 = trainData.reshape((-1,2048*40),order='F')
raw2 = valData.reshape((-1,2048*40),order='F')
raw3 = testData.reshape((-1,2048*40),order='F')

_1 = raw1.reshape((-1,2048))#,order='F')
_2 = raw2.reshape((-1,2048))#,order='F')
_3 = raw3.reshape((-1,2048))#,order='F')

print np.equal(raw1[0,0:2048],_1[0])
print np.equal(raw2[0,0:2048],_2[0])
print np.equal(raw3[0,0:2048],_3[0])
#'''

#p.p()
for i in range(40):
  #p.compare(_1[0+i:320*40:40],_1[320*40+i:640*40:40],_1[640*40+i:960*40:40],i)
#  p.compare(_1[0+i:320*40:40],_1[8*320*40+i:9*320*40:40],_1[16*320*40+i:17*320*40:40],i)
#  p.diff(_1[0+i:320*40:40],_1[320*40+i:640*40:40],_1[640*40+i:960*40:40],i)
#  p.std(_1[0+i:320*40:40],_1[320*40+i:640*40:40],_1[640*40+i:960*40:40],i)
#  p.toggle(_1[0+i:320*40:40,0:200],
#           _1[0+i:320*40:40,200:400],
#           _1[0+i:320*40:40,400:600],
#           _1[0+i:320*40:40,600:800],
#           _1[0+i:320*40:40,800:1000],
#           _1[0+i:320*40:40,1000:1200],
#           i)
#  p.show_fft(_1[0+i:320*40:40,0:],_1[320*40+i:640*40:40,0:],_1[640*40+i:960*40:40,0:],i)
  p.sameObject_diffR(_1[320*40*0+i:320*40*1:40],
                   _1[320*40*1+i:320*40*2:40],
                   _1[320*40*2+i:320*40*3:40],
                   _1[320*40*3+i:320*40*4:40],
                   _1[320*40*4+i:320*40*5:40],
                   _1[320*40*5+i:320*40*6:40],
                   _1[320*40*6+i:320*40*7:40],
                   _1[320*40*7+i:320*40*8:40],'s',i)
  p.sameObject_diffR(_1[320*40*8+i:320*40*9:40],
                   _1[320*40*9+i:320*40*10:40],
                   _1[320*40*10+i:320*40*11:40],
                   _1[320*40*11+i:320*40*12:40],
                   _1[320*40*12+i:320*40*13:40],
                   _1[320*40*13+i:320*40*14:40],
                   _1[320*40*14+i:320*40*15:40],
                   _1[320*40*15+i:320*40*16:40],'m',i)
  p.sameObject_diffR(_1[320*40*16+i:320*40*17:40],
                   _1[320*40*17+i:320*40*18:40],
                   _1[320*40*18+i:320*40*19:40],
                   _1[320*40*19+i:320*40*20:40],
                   _1[320*40*20+i:320*40*21:40],
                   _1[320*40*21+i:320*40*22:40],
                   _1[320*40*22+i:320*40*23:40],
                   _1[320*40*23+i:320*40*24:40],'l',i)

  p.diffObject_sameR(_1[320*40*0+i:320*40*1:40],
                   _1[320*40*8+i:320*40*9:40],
                   _1[320*40*16+i:320*40*17:40],'30',i)
  p.diffObject_sameR(_1[320*40*1+i:320*40*2:40],
                   _1[320*40*9+i:320*40*10:40],
                   _1[320*40*17+i:320*40*18:40],'40',i)
  p.diffObject_sameR(_1[320*40*2+i:320*40*3:40],
                   _1[320*40*10+i:320*40*11:40],
                   _1[320*40*18+i:320*40*19:40],'50',i)
  p.diffObject_sameR(_1[320*40*3+i:320*40*4:40],
                   _1[320*40*11+i:320*40*12:40],
                   _1[320*40*19+i:320*40*20:40],'60',i)
  p.diffObject_sameR(_1[320*40*4+i:320*40*5:40],
                   _1[320*40*12+i:320*40*13:40],
                   _1[320*40*20+i:320*40*21:40],'70',i)
  p.diffObject_sameR(_1[320*40*5+i:320*40*6:40],
                   _1[320*40*13+i:320*40*14:40],
                   _1[320*40*21+i:320*40*22:40],'80',i)
  p.diffObject_sameR(_1[320*40*6+i:320*40*7:40],
                   _1[320*40*14+i:320*40*15:40],
                   _1[320*40*22+i:320*40*23:40],'90',i)
  p.diffObject_sameR(_1[320*40*7+i:320*40*8:40],
                   _1[320*40*15+i:320*40*16:40],
                   _1[320*40*23+i:320*40*24:40],'100',i)

# check with original image 
'''
logs = ['data/001.log','data/002.log','data/003.log']
data = {}

start = time.time()
for i in range(len(logs)):
  data['{0}'.format(i)] = np.loadtxt(logs[i])
print time.time()-start
print data['0'].shape, data['1'].shape, data['2'].shape

trainData = np.concatenate((data['0'][0:320],data['1'][0:320],data['2'][0:320]),axis=0)
valData =  np.concatenate((data['0'][320:360],data['1'][320:360],data['2'][320:360]),axis=0)
testData =  np.concatenate((data['0'][360:400],data['1'][360:400],data['2'][360:400]),axis=0)

#trainData = trainData.reshape((-1,1,8192,40),order='F')
#valData = valData.reshape((-1,1,8192,40),order='F')
#testData = testData.reshape((-1,1,8192,40),order='F')
#print trainData.shape, valData.shape, testData.shape

print np.equal(raw1,trainData)
print np.equal(raw2,valData)
print np.equal(raw3,testData)
'''
