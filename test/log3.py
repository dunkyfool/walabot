import numpy as np
import h5py
import cv2
import time
import pandas as pd
from os import listdir, system
from os.path import isfile, join, isdir

def imgProcess(tmp_img,bk_img,thr,R):
    pass
    tmp_img = cv2.GaussianBlur(tmp_img, (51, 51), 0)
    bk_img = cv2.GaussianBlur(bk_img, (51, 51), 0)

    sub_img = cv2.absdiff(tmp_img,bk_img)
    thr_img = cv2.threshold(sub_img,thr,1,cv2.THRESH_BINARY)[1]

    res_img = cv2.resize(thr_img[200:600,200:600], (32, 32))
    res_img = res_img.reshape(1024)
    app_img = np.append(res_img,R) # append distance info
    return app_img.reshape(1,1024+1)
    # find crop image
    #cnts = cv2.findContours(thr_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    #for c in cnts:
    #  print  cv2.boundingRect(c)
    # show processed image
    #cv2.imshow('image1',thr_img[200:600,200:600])
    #cv2.imshow('image2',thr_img)
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
  paths = ['data/s30','data/s40','data/s50','data/s60','data/s70','data/s80','data/s90','data/s100',
           'data/m30','data/m40','data/m50','data/m60','data/m70','data/m80','data/m90','data/m100',
           'data/l30','data/l40','data/l50','data/l60','data/l70','data/l80','data/l90','data/l100']
  thres = [5,5,5,5,5,15,10,10,
           10,5,5,5,5,5,5,5,
           30,25,20,20,5,5,10,20]
  distR = [0.,1/7.,2/7.,3/7.,4/7.,5/7.,6/7.,1.]
  labels = []

  start = time.time()
  for idx in range(len(paths)):
    path = paths[idx]
    filename = sorted([f for f in listdir(path) if isfile(join(path,f))])
    for i in range(400): # retrieve only 400
      labels += [imgProcess(cv2.imread(join(path,filename[i]),0), cv2.imread(join(path,filename[-1]),0),thres[idx],distR[idx%8])]
  print time.time()-start

  labels = np.array(labels).reshape((-1,1024+1))
  #print labels.shape
  #print labels[200::400,1024]
  #raw_input()

  trainList = [ labels[i*400:i*400+320] for i in range(len(paths)) ]
  valList = [ labels[i*400+320:i*400+360] for i in range(len(paths)) ]
  testList = [ labels[i*400+360:(i+1)*400] for i in range(len(paths)) ]
  #print trainList[0].shape
  #print valList[0].shape
  #print testList[0].shape
  #raw_input()

  trainLabel = np.concatenate(trainList,axis=0)
  valLabel = np.concatenate(valList,axis=0)
  testLabel = np.concatenate(testList,axis=0)
  #print trainLabel.shape
  #print valLabel.shape
  #print testLabel.shape
  #raw_input()
  #print trainLabel.shape, valLabel.shape,testLabel.shape
  # build training label & empty data
  with h5py.File('data/tri_walabot.h5', 'w') as hf:
      hf.create_dataset('trainData',(trainLabel.shape[0],1,2048,40),compression='gzip',maxshape=(None,None,None,None))
      hf.create_dataset('trainLabel',data=trainLabel,compression='gzip',maxshape=(None,None))
      hf.create_dataset('valData',(valLabel.shape[0],1,2048,40),compression='gzip',maxshape=(None,None,None,None))
      hf.create_dataset('valLabel',data=valLabel,compression='gzip',maxshape=(None,None))
      hf.create_dataset('testData',(testLabel.shape[0],1,2048,40),compression='gzip',maxshape=(None,None,None,None))
      hf.create_dataset('testLabel',data=testLabel,compression='gzip',maxshape=(None,None))
  del trainList,valList,testList,trainLabel,valLabel,testLabel
  print "Load time: ", (time.time()-convert_start)
  #'''


  # txt to data
  logs = ['data/s30.log','data/s40.log','data/s50.log','data/s60.log','data/s70.log','data/s80.log','data/s90.log','data/s100.log',
           'data/m30.log','data/m40.log','data/m50.log','data/m60.log','data/m70.log','data/m80.log','data/m90.log','data/m100.log',
           'data/l30.log','data/l40.log','data/l50.log','data/l60.log','data/l70.log','data/l80.log','data/l90.log','data/l100.log']
  #with h5py.File('data/tri_walabot.h5', 'w') as hf:
  #    hf.create_dataset('trainData',(7680,1,2048,40),compression='gzip')
  #    hf.create_dataset('valData',(960,1,2048,40),compression='gzip')
  #    hf.create_dataset('testData',(960,1,2048,40),compression='gzip')

  for idx in range(len(logs)):
    start = time.time()
    data = {}
    #data['{0}'.format(i)] = np.loadtxt(logs[i])
    data['{0}'.format(idx)] = np.genfromtxt(logs[idx])
    print time.time()-start

    # feature 8192 * 40
    trainData =  data['{0}'.format(idx)][0:320]
    valData =  data['{0}'.format(idx)][320:360]
    testData =  data['{0}'.format(idx)][360:400]

    # transform
    trainData = trainData.reshape((-1,1,8192,40),order='F')
    valData = valData.reshape((-1,1,8192,40),order='F')
    testData = testData.reshape((-1,1,8192,40),order='F')
    #print trainData.shape, valData.shape, testData.shape

    # shorten from 8192 to 2048
    trainData = trainData[:,:,0:2048,:]
    valData = valData[:,:,0:2048,:]#.reshape((-1,1,8192,40),order='F')
    testData = testData[:,:,0:2048,:]#.reshape((-1,1,8192,40),order='F')
    #print 'shorten version:',trainData.shape, valData.shape, testData.shape

    # build training set
    with h5py.File('data/tri_walabot.h5', 'a') as hf:
      hf['trainData'][idx*320:(idx+1)*320] = trainData
      hf['valData'][idx*40:(idx+1)*40] = valData
      hf['testData'][idx*40:(idx+1)*40] = testData

    #with h5py.File('data/tri_walabot.h5','r') as hf:
    #  a = np.array(hf['trainData'])
    #  print a[0:320].max();print a[320:640].max();print a.shape
    #  a = np.array(hf['valData'])
    #  print a[0:40].max();print a[40:80].max();print a.shape
    #  a = np.array(hf['testData'])
    #  print a[0:40].max();print a[40:80].max();print a.shape

    del data,trainData,valData,testData
  print "Load time: ", (time.time()-convert_start)

def load():
    start = time.time()
    trainData,trainLabel,valData,valLabel,testData,testLabel = None,None,None,None,None,None
    with h5py.File('data/tri_walabot.h5','r') as hf:
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
