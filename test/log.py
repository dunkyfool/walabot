import numpy as np
import h5py
import cv2
import time
from os import listdir, system
from os.path import isfile, join, isdir

def imgProcess(tmp_img,bk_img,thr):
    pass
    tmp_img = cv2.GaussianBlur(tmp_img, (51, 51), 0)
    bk_img = cv2.GaussianBlur(bk_img, (51, 51), 0)

    sub_img = cv2.absdiff(tmp_img,bk_img)
    thr_img = cv2.threshold(sub_img,thr,1,cv2.THRESH_BINARY)[1]

    res_img = cv2.resize(thr_img[10:350,202:374], (32, 32))
    return res_img.reshape(1,1024)
    # find crop image
    #cnts = cv2.findContours(thr_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    #for c in cnts:
    #  print  cv2.boundingRect(c)
    # show processed image
    #cv2.imshow('image1',thr_img[10:350,202:374])
    #cv2.imshow('image2',res_img)
    #k = cv2.waitKey(0) & 0xFF
    #if k == 27:         # wait for ESC key to exit
    #  cv2.destroyAllWindows()


def convert():
  pass
  # Image process
  # 1. List all data (sorted)
  # 2. Remove background (absdiff/threshold)
  # 3. crop ROI
  # 4. resize (32x32)
  # 5. Divide train/val/test label

  # image to label
  convert_start = time.time()
  #'''
  paths = ['data/001/','data/002','data/003']
  thres = [3,20,12]
  labels = []

  start = time.time()
  for idx in range(len(paths)):
    path = paths[idx]
    filename = sorted([f for f in listdir(path) if isfile(join(path,f))])
    for i in range(400): # retrieve only 400
      labels += [imgProcess(cv2.imread(join(path,filename[i]),0), cv2.imread(join(path,filename[-1]),0),thres[idx])]
  print time.time()-start

  labels = np.array(labels).reshape((-1,1024))
  #print labels.shape

  trainLabel = np.concatenate((labels[0:320],labels[400:720],labels[800:1120]),axis=0)
  valLabel = np.concatenate((labels[320:360],labels[720:760],labels[1120:1160]),axis=0)
  testLabel = np.concatenate((labels[360:400],labels[760:800],labels[1160:1200]),axis=0)
  #print trainLabel.shape, valLabel.shape,testLabel.shape
  #'''

  # txt to data
  logs = ['data/001.log','data/002.log','data/003.log']
  data = {}

  start = time.time()
  for i in range(len(logs)):
    data['{0}'.format(i)] = np.loadtxt(logs[i])
  print time.time()-start
  print data['0'].shape, data['1'].shape, data['2'].shape

  # feature 8192 * 40
  trainData = np.concatenate((data['0'][0:320],data['1'][0:320],data['2'][0:320]),axis=0)
  valData =  np.concatenate((data['0'][320:360],data['1'][320:360],data['2'][320:360]),axis=0)
  testData = np.concatenate((data['0'][360:400],data['1'][360:400],data['2'][360:400]),axis=0)

  # transform
  trainData = trainData.reshape((-1,1,8192,40),order='F')
  valData = valData.reshape((-1,1,8192,40),order='F')
  testData = testData.reshape((-1,1,8192,40),order='F')
  print trainData.shape, valData.shape, testData.shape

  # build training set
  #'''
  with h5py.File('data/walabot.h5', 'w') as hf:
      hf.create_dataset('trainData',data=trainData,compression='gzip')
      hf.create_dataset('trainLabel',data=trainLabel,compression='gzip')
      hf.create_dataset('valData',data=valData,compression='gzip')
      hf.create_dataset('valLabel',data=valLabel,compression='gzip')
      hf.create_dataset('testData',data=testData,compression='gzip')
      hf.create_dataset('testLabel',data=testLabel,compression='gzip')
  #'''
  print "Load time: ", (time.time()-convert_start)

def load():
    start = time.time()
    trainData,trainLabel,valData,valLabel,testData,testLabel = None,None,None,None,None,None
    with h5py.File('data/walabot.h5','r') as hf:
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
  #convert()
  #trainData,trainLabel,valData,valLabel,testData,testLabel = load()
