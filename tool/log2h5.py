import numpy as np
import h5py
import time

def convert():

    start = time.time()
    data = np.loadtxt('data/walabot.log')
    #data = np.genfromtxt('walabot.log')
    print data.shape

    # Num train 80%, val 10%, test 10%
    total = data.shape[0]
    trainNum = int(total*0.8)
    valNum = int((total-trainNum) * 0.5)
    testNum = total-trainNum-valNum
    print trainNum, valNum, testNum

    # Index
    trainflag = trainNum
    valflag = trainNum + valNum

    trainData, trainLabel = data[0:trainflag,0:-2], data[0:trainflag,-2:]
    valData, valLabel = data[trainflag:valflag,0:-2], data[trainflag:valflag,-2:]
    testData, testLabel = data[valflag:,0:-2], data[valflag:,-2:]

    print trainData.shape, trainLabel.shape
    print valData.shape, valLabel.shape
    print testData.shape, testLabel.shape

    with h5py.File('data/walabot.h5', 'w') as hf:
      hf.create_dataset('trainData',data=trainData,compression='gzip')
      hf.create_dataset('trainLabel',data=trainLabel,compression='gzip')
      hf.create_dataset('valData',data=valData,compression='gzip')
      hf.create_dataset('valLabel',data=valLabel,compression='gzip')
      hf.create_dataset('testData',data=testData,compression='gzip')
      hf.create_dataset('testLabel',data=testLabel,compression='gzip')


    print "Load time: ", (time.time()-start)

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
    return trainData, trainLabel, valData, valLabel, testData, testLabel

if __name__=='__main__':
    pass
    convert()
    trainData,trainLabel,valData,valLabel,testData,testLabel = load()
