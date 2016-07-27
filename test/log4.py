import numpy as np
import h5py
import cv2
import time
import pandas as pd
import os
from os import listdir, system
from os.path import isfile, join, isdir

def imgProcess(tmp_img,bk_img,thr,cls,x,y,z,ver,opt=False):
    pass
    # remove background & build silhouette
    tmp_img = cv2.GaussianBlur(tmp_img, (51, 51), 0)
    bk_img = cv2.GaussianBlur(bk_img, (51, 51), 0)
    sub_img = cv2.absdiff(tmp_img,bk_img)

    if not opt: # build label
      thr_img = cv2.threshold(sub_img,thr,1,cv2.THRESH_BINARY)[1]
    else:       # build threshold (easy to review on picture)
      print 'threshold', thr
      thr_img = cv2.threshold(sub_img,thr,255,cv2.THRESH_BINARY)[1]

    # resize image to expected size
    res_img = cv2.resize(thr_img, (128, 128))
    res_img = res_img.reshape(128*128)

    if not opt: # build label
        app_img = np.append(res_img,[cls,x,y,z,ver]) # append other info
        return app_img.reshape(1,128*128+cls.shape[0]+3+1)

    else:
        buff = [] # save threshold number
        tmp = str() # string to int
        while True:
            cv2.imshow('image2',thr_img)
            #cv2.imshow('image2',res_img)

            k = cv2.waitKey(1) & 0xFF
            if k == 115:         # s: ok next one
              cv2.destroyAllWindows()
              return True, thr

            if k == 141:         # enter: reset tresh
              cv2.destroyAllWindows()

              if buff:           # buff not empty
                for w in buff:
                  tmp += str(w)
                thr = int(tmp)
              else: thr = 0

              return False, thr

            if k >=176 and k <=185: # number: new thresh
              buff += [k-176]

            if k != 255: print k

def findThres(path):
  pass
  # 1. Load all folder(s) or load all log(s)
  # 2. Extract first picture as example
  # 3. Use imgProcess to review 

  ############
  # Variable #
  ############
  folders = sorted([f for f in listdir(path) if isdir(join(path,f))])
  folders.remove('a__name')
  folders.remove('archive')
  folders.remove('tmp')
  #_folders = np.array(folders).reshape(-1,3)
  #print _folders;print len(folders);raw_input()
  #print logs,folders

  #######################
  # Fine Tune Threshold #
  #######################
  record = [] # save threshold
  idx = 0
  thr = 0

  while idx < len(folders):
    print idx
    first_pic = sorted(listdir(join(path,folders[idx])))

    if not first_pic: print '['+str(idx)+']', 'No image in this folder'
    else: first_pic = first_pic[0]

    #thr = raw_input('Enter threshold: ')
    #if not thr: thr = 0

    print join(path,folders[idx],first_pic)
    outcome,thr = imgProcess(cv2.imread(join(path,folders[idx],first_pic),0),
                         cv2.imread('data/background.jpg',0),thr,0,0,0,0,0,opt=True)
    if outcome: # True: save threshold and go to next one
      record += [thr]
      idx += 1

  ################
  # Save to file #
  ################
  record = np.array(record)
  print record, record.shape

  _ = raw_input('Overwrite[Y/n]: ')
  if _ == 'y' or _ == 'Y': os.system('rm data/thre.h5')

  with h5py.File('data/thre.h5', 'w') as hf:
      hf.create_dataset('threshold',data=record,compression='gzip')

  _checkthr() # verify

def _checkthr():
  with h5py.File('data/thre.h5') as hf:
    thr = np.array(hf['threshold'])
  print thr; print thr.shape

def convert_img2label():
  pass
  # Image process
  # 1. List all data (sorted)
  # 2. Remove background (absdiff/threshold)
  # 3. resize (128x128)
  # 4. Divide train/val label

  start = time.time()
  ############
  # Variable #
  ############
  paths = sorted([join('data/',f) for f in listdir(path) if isdir(join('data',f))])

  # remove redundent folder(s)
  paths.remove('data/a__name')
  paths.remove('data/archive')
  paths.remove('data/tmp')

  # load fine-tunning threshold
  with h5py.File('data/thre.h5','r') as hf:
    thres = np.array(h5['threshold'])
  clsList = []

  # search class num & build matrix
  for d in paths:
    name, x, y, z, ver = d.split('/')[-1].split('_')
    if name not in clsList: clsList += [name]
  cls = np.eye(len(clsList))

  ###########################
  # Initialize HDF5 dataset #
  ###########################
  _ = raw_input('Overwrite[y/N]: ')
  if _ == 'y' or _ == 'Y':
    with h5py.File('data/tri_walabot.h5', 'w') as hf:
      hf.create_dataset('trainLabel',(0,0),compression='gzip',maxshape=(None,None))
      hf.create_dataset('valLabel',(0,0),compression='gzip',maxshape=(None,None))

  ###############
  # Build Label #
  ###############
  # data/clay_30_0_0_v1/000000001.jpg
  # |>      path      <|
  #                    |> filename <|
  for idx in range(len(paths)):
    labels = [] # reset
    path = paths[idx]
    name,x,y,z,ver = path.split('/')[-1].split('_')
    filename = sorted([f for f in listdir(path) if isfile(join(path,f))])

    for pic in filename:
      labels += [imgProcess(cv2.imread(join(path,pic),0),
                            cv2.imread('data/background',0),thres[idx],
                            cls[clsList.index(name)],x,y,z,int(ver[1:]))]

    labels = np.array(labels).reshape((-1,128*128+len(clsList)+3+1))

    # save to hdf5 dataset
    with h5py.File('data/tri_walabot.h5', 'a') as hf:
      # find current size of dataset & calculate new one
      trainCurrSize = hf['trainLabel'].shape[0]
      valCurrSize = hf['valLabel'].shape[0]
      trainNum = int(labels.shape[0]*0.9)
      valNum = labels.shape[0] - trainNum

      trainShape = (trainCurrSize + trainNum, labels.shape[1])
      valShape = (valCurrSize + valNum, labels.shape[1])

      hf['trainLabel'].resize((trainShape))
      hf['valLabel'].resize((valShape))

      hf['trainLabel'][trainCurrSize:trainCurrSize+trainNum] = labels[0:trainNum]
      hf['valLabel'][valCurrSize:valCurrSize+valNum] = labels[trainNum:]

    del labels

  print time.time()-start

def convert_log2data():
  ############
  # Variable #
  ############
  logs = [join('data',f) for f in listdir('data') if isfile(join('data',f)) and f[-4:]=='.log']

  ###########################
  # Initialize HDF5 dataset #
  ###########################
  _ = raw_input('Overwrite[y/N]: ')
  if _ == 'y' or _ == 'Y':
    with h5py.File('data/tri_walabot.h5', 'w') as hf:
      hf.create_dataset('trainData',(0,0,0,0),compression='gzip',maxshape=(None,None,None,None))
      hf.create_dataset('valData',(0,0,0,0),compression='gzip',maxshape=(None,None,None,None))

  ###############
  # log to data #
  ###############
  for idx in range(len(logs)):
    start = time.time()
    data = np.genfromtxt(logs[idx])
    print time.time()-start

    with h5py.File('data/tri_walabot.h5', 'a') as hf:
        # find current size of dataset & calculate new one
        trainCurrSize = hf['trainData'].shape[0]
        valCurrSize = hf['valData'].shape[0]
        trainNum = int(data.shape[0]*0.9)
        valNum = data.shape[0] - trainNum

        trainShape = (trainCurrSize + trainNum, data.shape[1],data.shape[2],data.shape[3])
        valShape = (valCurrSize + valNum, data.shape[1],data.shape[2],data.shape[3])

        hf['trainData'].resize((trainShape))
        hf['valData'].resize((valShape))

        # feature 8192 * 40
        trainData =  data[0:trainNum]
        valData =  data[trainNum:]

        # transform
        trainData = trainData.reshape((-1,1,8192,40),order='F')
        valData = valData.reshape((-1,1,8192,40),order='F')
        #print trainData.shape, valData.shape, testData.shape

        # shorten from 8192 to 2048
        trainData = trainData[:,:,0:2048,:]
        valData = valData[:,:,0:2048,:]#.reshape((-1,1,8192,40),order='F')
        #print 'shorten version:',trainData.shape, valData.shape, testData.shape

        hf['trainData'][trainCurrSize:trainCurrSize+trainNum] = trainData
        hf['valData'][valCurrSize:valCurrSize+valNum] = valData

        #with h5py.File('data/tri_walabot.h5','r') as hf:
        #  a = np.array(hf['trainData'])
        #  print a[0:320].max();print a[320:640].max();print a.shape
        #  a = np.array(hf['valData'])
        #  print a[0:40].max();print a[40:80].max();print a.shape
        #  a = np.array(hf['testData'])
        #  print a[0:40].max();print a[40:80].max();print a.shape

    del data,trainData,valData

def convert():
  convert_img2label()
  convert_log2data()

def load():
    start = time.time()
    trainData,trainLabel,valData,valLabel,testData,testLabel = None,None,None,None,None,None

    with h5py.File('data/tri_walabot.h5','r') as hf:
        trainData = np.array(hf['trainData'])
        trainLabel = np.array(hf['trainLabel'])
        valData = np.array(hf['valData'])
        valLabel = np.array(hf['valLabel'])
        #testData = np.array(hf['testData'])
        #testLabel = np.array(hf['testLabel'])

    print 'trainData',trainData.shape
    print 'trainLabel',trainLabel.shape
    print 'valData',valData.shape
    print 'valLabel',valLabel.shape
    #print 'testData',testData.shape
    #print 'testLabel',testLabel.shape
    print 'laod time: ', (time.time()-start)

    print trainData.max(),trainData.min(),trainData.mean()
    print trainLabel.max(),trainLabel.min(),trainLabel.mean()

    return trainData, trainLabel, valData, valLabel, testData, testLabel

if __name__=='__main__':
  pass
  findThres('data')
  #convert()
  #trainData,trainLabel,valData,valLabel = load()
  #trainData,trainLabel,valData,valLabel,testData,testLabel = load()
