from model.NNWalabot3 import *
from tool.log2 import *
import time
import matplotlib.pyplot as plt
import sys
import os
import cv2

def quick_scan(X,y,X1,y1,e,lr_range=[-3.0,-3.7],reg_range=[1,0],epoch=1,sample=10):
  results = {}
  learning_rates = lr_range
  regularization_strengths = reg_range
  #best_val = -1
  best_val = np.inf
  best_lr, best_reg = None, None

  tic = time.time()
  for i in range(sample):
    print '['+str(i)+']'
    # random choose lr & reg within the range
    lr = 10**np.random.uniform(learning_rates[0],learning_rates[1])
    reg = 10**np.random.uniform(regularization_strengths[0],regularization_strengths[1])
    print 'lr:\t'+str(lr)
    print 'reg:\t'+str(reg)

    net = modelX()
    net.loss(X,y,X1,y1,e,mode='train',lr=lr,reg=reg,batch=10,epoch=epoch)
    #results[(lr,reg)]=(net.X_acc_history[-1],net.X1_acc_history[-1])
    #results[(lr,reg)]=(net.X_loss_history[-1],net.X1_loss_history[-1])
    cmd = 'echo '+str(lr)+' '+str(reg)+' '+str(net.X_acc_history[-1])+' '+str(net.X1_acc_history[-1])+'>>qs.log'
    os.system(cmd)
    if best_val > net.X1_acc_history[-1]:
      best_val = net.X1_acc_history[-1]
      best_lr = lr
      best_reg = reg
    # shrink mem
    #del net

  toc = time.time()
  print 'Total Training: computed in %fs' % (toc - tic)
  print 'Best Validation Record %.5f' % (best_val)
  print 'Best Validation learning rate %.10f' % (best_lr)
  print 'Best Validation regularization %.10f' % (best_reg)


def quick_scan_plot():
  ##########################################
  # Visualize the cross-validation results #
  ##########################################
  import math

  data=None
  results={}
  data = np.loadtxt('qs.log')
  for i in range(data.shape[0]):
    results[(data[i,0],data[i,1])] = (data[i,2],data[i,3])

  x_scatter = [math.log10(x[0]) for x in results]
  y_scatter = [math.log10(x[1]) for x in results]

  print x_scatter,y_scatter

  # plot training accuracy
  marker_size = 100
  colors = [results[x][0] for x in results]
  plt.subplot(2, 1, 1)
  plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
  plt.colorbar()
  plt.xlabel('log learning rate')
  plt.ylabel('log regularization strength')
  plt.title('Training Accuracy')

  # plot validation accuracy
  colors = [results[x][1] for x in results] # default size of markers is 20
  plt.subplot(2, 1, 2)
  plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
  plt.colorbar()
  plt.xlabel('log learning rate')
  plt.ylabel('log regularization strength')
  plt.title('Validation Accuracy')
  plt.savefig("loss.png")
  plt.show()

def marathon(X,y,X1,y1,X2,y2,e,lr=1e-4,reg=1e-4,epoch=20):
  net = modelX()
  #'''
  opt = raw_input('Restart training??[y/n]')
  if opt=='y':
    net.loss(X,y,X1,y1,e,mode='train',lr=lr,reg=reg,batch=10,epoch=epoch)
  elif opt=='n':
    net.loss(X,y,X1,y1,e,mode='train',lr=lr,reg=reg,batch=10,epoch=epoch,opt=False)
  #'''
  #_net = modelX()
  #_net.loss(X2,y2,X1,y1,mode='test')

  ####################################################
  # Visualize training loss and train / val accuracy #
  ####################################################
  plt.subplot(2, 1, 1)
  plt.title('Training loss')
  plt.plot(net.X_loss_history, 'o')
  plt.xlabel('Iteration')

  plt.subplot(2, 1, 2)
  plt.title('Accuracy')
  plt.plot(net.X_acc_history, '-o', label='train')
  plt.plot(net.X1_acc_history, '-o', label='val')
  plt.plot([0.5] * len(net.X1_acc_history), 'k--')
  plt.xlabel('Epoch')
  plt.legend(loc='lower right')
  plt.gcf().set_size_inches(15, 12)
  plt.savefig("graph.png")
  plt.show()
  pass

def review(X,y,X1,y1,e):
  net = modelX()
  _,_ = net.loss(X,y,X1,y1,e,mode='test')

def duel(X,y,X1,y1,e):
  net = modelX()
  org_output,org_output1 = net.loss(X,y,X1,y1,e,mode='test')
  #for i in range(1024):
  #  if y[0,i]!=1 and y[0,i]!=0:
  #    print y[0,i]
  #raw_input('pause')
  for _ in range(org_output.shape[0]):
    idx = int(np.random.rand()* org_output.shape[0])
    #print idx
    tmp = np.ceil(org_output[idx] * (org_output[idx]>0.5))
    #tmp = org_output[idx]
#    for i in range(1024):
#      if tmp[i]!=1 and tmp[i]!=0:
#        print i, tmp[i]
#    raw_input('pause')
    pred_pic = tmp.reshape(32,32)*255
    corr_pic = y[idx].reshape(32,32)*255

    plt.clf()
    plt.subplot(121)
    plt.imshow(pred_pic,cmap='Greys_r')
    plt.subplot(122)
    plt.imshow(corr_pic,cmap='Greys_r')

    plt.savefig('data/compare_img/{0}.png'.format(idx))

def mask_input(X,y,X1,y1,e):
  idx = 800#int(sys.argv[1])
  #print idx,type(idx)
  span = 3
  #mask_num = 1024
  #interval = X.shape[2]/mask_num

  #mask = np.zeros((1,1,mask_num,40))

  net = modelX()
  inp = X[idx:idx+3,:,:,:].reshape((span/1,1,2048,40))
  oup = y[idx:idx+3].reshape((span/1,-1))
  #inp = X[idx::-40,:,:,:].reshape((span/1,1,2048,40))
  #oup = y[idx::-40].reshape((span/1,-1))
  #print input[:,:,mask_num*i:mask_num*(i+1),:].shape
  #raw_input()
 # input[:,:,mask_num*i:mask_num*(i+1),:] = mask
  #print input.shape
  #print output.shape
  #raw_input()
  pred, _ = net.loss(X,y,inp,oup,e,mode='test')
  #'''
  tmp = np.ceil(pred * (pred>0.5))
  pred_pic = tmp.reshape(3,32,32)*255
  corr_pic = oup.reshape(3,32,32)*255

  plt.clf()
  plt.subplot(231)
  plt.imshow(pred_pic[0],cmap='Greys_r')
  plt.subplot(232)
  plt.imshow(pred_pic[1],cmap='Greys_r')
  plt.subplot(233)
  plt.imshow(pred_pic[2],cmap='Greys_r')
  plt.subplot(234)
  plt.imshow(corr_pic[0],cmap='Greys_r')
  plt.subplot(235)
  plt.imshow(corr_pic[1],cmap='Greys_r')
  plt.subplot(236)
  plt.imshow(corr_pic[2],cmap='Greys_r')

  plt.savefig('data/mask_input/{0}.png'.format(0))
  #'''
if __name__=='__main__':
  os.system('cp tmp tmp.bak && echo > tmp')
  e = mp.Event()
  e2 = mp.Event()
  prev_cpu = mp.Value('d',0.0)
  prev_mem = mp.Value('d',0.0)
  prev_swap = mp.Value('d',0.0)
  ctr = mp.Value('d',0.0)
  state = mp.Value('d',0.0)

  p1 = mp.Process(target=log, args=(e,e2,prev_cpu,prev_mem,prev_swap,ctr,state))
  p1.start()
  start = time.time()
  trainData,trainLabel,valData,valLabel,testData,testLabel = load()
  #print testData.shape, testLabel.shape
  print time.time()-start

  # speedup test
  #testData = testData[0:100,:]
  #testLabel = testLabel[0:100,:]

  #trainData = trainData.reshape(-1,32,32,3)
  #valData = valData.reshape(-1,32,32,3)
  #testData = testData.reshape(-1,32,32,3)

  trainData = trainData.astype(np.float)
  valData = valData.astype(np.float)
  testData = testData.astype(np.float)

  '''
  trainData *= 6.
  valData *= 6.
  testData *= 6.
  '''

  #trainData = trainData-np.mean(trainData,axis=0)
  #valData = valData-np.mean(trainData,axis=0)
  #testData = testData-np.mean(trainData,axis=0)

  print 'max',trainData.max(),'min',trainData.min(), 'mean',trainData.mean()
  print 'max',trainLabel.max(),'min',trainLabel.min(), 'mean',trainLabel.mean()
  '''
  ctr0,ctr1,ctrx= 0,0,0
  for i in trainLabel[640]:
    if i==0: ctr0+=1
    elif i==1: ctr1+=1
    else: ctrx+=1
  print ctr0,ctr1,ctrx,ctr0+ctr1+ctrx
  '''
  start = time.time()
#  quick_scan_plot()
#  quick_scan(trainData[0:960],trainLabel[0:960],
#             valData,valLabel,e,lr_range=[-2,-3],reg_range=[-2,-3],epoch=10,sample=1)
#  marathon(trainData,trainLabel,valData,valLabel,testData,testLabel,e,
#           lr=0.00138100536852,reg=0.00140667975999, epoch=10)
#  review(testData,testLabel,valData,valLabel,e)
#  duel(testData,testLabel,valData,valLabel,e)
#  mask_input(testData,testLabel,valData,valLabel,e)
  e2.set()
  print time.time()-start
  pass
