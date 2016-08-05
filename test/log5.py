import numpy as np
import h5py
import time
import os
from os import listdir, system
from os.path import isfile, join, isdir
import sys,select

###################
# Global Variable #
###################
path = 'antenna'
logName = '2D_location.h5'
MODE = 'test'

def timeout(string):
  print 'Skip in FIVE seconds',string
  i,o,e = select.select([sys.stdin],[],[],5)
  if i:
    return sys.stdin.readline().strip()

def init(mode):
  if mode == 'train':
      with h5py.File(join(path,logName),'a') as hf:
          hf['trainLabel'].resize((0,0))
          hf['trainData'].resize((0,0,0,0))
  elif mode == 'val':
      with h5py.File(join(path,logName),'a') as hf:
          hf['valLabel'].resize((0,0))
          hf['valData'].resize((0,0,0,0))
  elif mode == 'test':
      with h5py.File(join(path,logName),'a') as hf:
          hf['testLabel'].resize((0,0))
          hf['testData'].resize((0,0,0,0))
  else:
      with h5py.File(join(path,logName),'w') as hf:
          hf.create_dataset('trainLabel',(0,0),compression='gzip',maxshape=(None,None))
          hf.create_dataset('valLabel',(0,0),compression='gzip',maxshape=(None,None))
          hf.create_dataset('testLabel',(0,0),compression='gzip',maxshape=(None,None))
          hf.create_dataset('trainData',(0,0,0,0),compression='gzip',maxshape=(None,None,None,None))
          hf.create_dataset('valData',(0,0,0,0),compression='gzip',maxshape=(None,None,None,None))
          hf.create_dataset('testData',(0,0,0,0),compression='gzip',maxshape=(None,None,None,None))

def convert_log2data(mode):
  ############
  # Variable #
  ############
  logs = [join(path,f) for f in listdir(path) if f[-4:]=='.txt']
  dataName = mode + 'Data'

  ###############
  # log to data #
  ###############
  data = np.zeros((len(logs),2048))
  for idx in range(len(logs)):
    start = time.time()
    data[idx] = np.genfromtxt(logs[idx])[:,1]

  data =  np.swapaxes(data,0,1)
  data = data.reshape((1,1,2048,len(logs)))

  with h5py.File(join(path,logName), 'a') as hf:
    # find current size of dataset & calculate new one
    CurrSize = hf[dataName].shape[0]
    Shape = (CurrSize + 1,1,2048,len(logs))
    hf[dataName].resize((Shape))
    hf[dataName][CurrSize:CurrSize+1] = data

  _check_log2data(mode)

def _check_log2data(mode):
  v = None
  dataName = mode + 'Data'
  with h5py.File(join(path,logName),'r') as hf:
    v = np.array(hf[dataName])
  print v.shape
  print v.max()
  print v.min()

def convert_label(mode,d,w):
  d = (d-30) / 70.
  w = w / 40.
  data = np.array([d,w])
  labelName = mode + 'Label'

  with h5py.File(join(path,logName),'a') as hf:
    CurrSize = hf[labelName].shape[0]
    Shape = (CurrSize+1,2)
    hf[labelName].resize((Shape))
    hf[labelName][CurrSize:CurrSize+1] = data

  _check_label(mode)

def _check_label(mode):
  v = None
  labelName = mode + 'Label'
  with h5py.File(join(path,logName),'r') as hf:
    v = np.array(hf[labelName])
  print v

def load():
    start = time.time()
    trainData,trainLabel,valData,valLabel,testData,testLabel = None,None,None,None,None,None

    with h5py.File(join(path,logName),'r') as hf:
        trainData = np.array(hf['trainData'])
        trainLabel = np.array(hf['trainLabel'])
        valData = np.array(hf['valData'])
        valLabel = np.array(hf['valLabel'])
        testData = np.array(hf['testData'])
        testLabel = np.array(hf['testLabel'])

    print 'trainData',trainData.shape
    print 'trainLabel',trainLabel.shape
    print 'valData',valData.shape
    print 'valLabel',valLabel.shape
    print 'testData',testData.shape
    print 'testLabel',testLabel.shape
    print 'laod time: ', (time.time()-start)

    print trainData.max(),trainData.min(),trainData.mean()
    print trainLabel.max(),trainLabel.min(),trainLabel.mean()

    return trainData, trainLabel, valData, valLabel, testData, testLabel

if __name__=='__main__':
  pass

  os.system("clear")
  #print "Create 2D_location.h5 dataset!!"
  #print "Current Mode [", MODE.upper(),"]"
  #depth = raw_input("Enter Depth(cm)[30-100]: ")
  #width = raw_input("Enter Width(cm)[ 0-40 ]: ")

  ############################
  ## Initialize HDF5 dataset #
  ############################
  #if not isfile(join(path,logName)):
  #  init(0)

  #_ = timeout('Overwrite[y/N]: ')
  #if _ == 'y' or _ == 'Y':
  #  init(MODE)

  with h5py.File(join(path,logName),'r') as hf:
    print '====== Status of Current Dataset ======'
    for k in hf.keys():
      print k.upper(),'\t',hf[k].shape
    print '\n'

  ## execute SensorSampleCode.cpp
  #cmd = './buildAll.sh'
  #os.system(cmd)

  #convert_log2data(MODE)
  #convert_label(MODE,int(depth),int(width))

  #trainData,trainLabel,valData,valLabel,testData,testLabel = load()
