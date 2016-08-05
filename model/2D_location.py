import numpy as np
import tensorflow as tf
import multiprocessing as mp
from utils.layers import *
from utils.initParams import *
from utils.monitor2 import *
# All DNN w/o softmax
# CNN - CNN1/2 - CNN - CNN1/2 - CNN - CNN1/2 - CNN (7)
# DNN-10213x1396 DNN-1396x1396 DNN-1396x1024 (3)
# !! Reduce CNN to 5 or 6 layers

####################
# Random Minibatch #
####################
def random_minibatch(data,label,size):
  """
  Random choose minibatch from data and corresponding label
  * size : minibatch size
  """
  mask = [np.arange(size,data.shape[0])]
  new_data = np.zeros_like(data)
  new_data = np.delete(new_data,mask,axis=0)
  new_label = np.zeros_like(label)
  new_label = np.delete(new_label,mask,axis=0)

  for i in range(size):
    idx = np.floor(np.random.uniform(low=0,high=data.shape[0])).astype(np.int64)
    new_data[i] += data[idx]
    new_label[i] += label[idx]
  return new_data, new_label

def _log(opt,e,cmd):
  if opt:
    os.system(cmd)
    e.set()
  else:
    os.system(cmd)
    e.clear()

class modelX():
  def __init__(self):
    ############
    # Variable #
    ############
    # x: image input
    # y_hat: image label
    # f: cnn filter weight
    # fb: cnn filter bias
    # w: fc weight
    # b: fc bias
    # conv_para: cnn stride padding
    # pool_para: maxpool stride padding kernel
    # regularizer: L2 loss

    # Input & label
    self.x = tf.placeholder(tf.float32,[None,1,2048,13])
    self.y_hat = tf.placeholder(tf.float32,[None, 2])

    # CNN-1
    self.f1 = weight([1,3,13,13])
    self.fb1 = bias([13])
    self.bng1 = gamma([13])
    self.bnb1 = beta([13])
    self.bnm1 = beta([13])
    self.bnv1 = beta([13])

    # CNN-2 (pooling)
    self.half_f1 = weight([1,4,13,13])
    self.half_fb1 = bias([13])
    self.half_bng1 = gamma([13])
    self.half_bnb1 = beta([13])
    self.half_bnm1 = beta([13])
    self.half_bnv1 = beta([13])

    # CNN-3
    self.f2 = weight([1,3,13,13])
    self.fb2 = bias([13])
    self.bng2 = gamma([13])
    self.bnb2 = beta([13])
    self.bnm2 = beta([13])
    self.bnv2 = beta([13])

    # CNN-4 (pooling)
    self.half_f2 = weight([1,4,13,13])
    self.half_fb2 = bias([13])
    self.half_bng2 = gamma([13])
    self.half_bnb2 = beta([13])
    self.half_bnm2 = beta([13])
    self.half_bnv2 = beta([13])

    # CNN-5
    self.f3 = weight([1,3,13,13])
    self.fb3 = bias([13])
    self.bng3 = gamma([13])
    self.bnb3 = beta([13])
    self.bnm3 = beta([13])
    self.bnv3 = beta([13])

    # CNN-6 (pooling)
    self.half_f3 = weight([1,4,13,13])
    self.half_fb3 = bias([13])
    self.half_bng3 = gamma([13])
    self.half_bnb3 = beta([13])
    self.half_bnm3 = beta([13])
    self.half_bnv3 = beta([13])

    # CNN-7
    self.f4 = weight([1,3,13,13])
    self.fb4 = bias([13])
    self.bng4 = gamma([13])
    self.bnb4 = beta([13])
    self.bnm4 = beta([13])
    self.bnv4 = beta([13])

    # FC-1
    self.w1 = weight([256*13,2048])
    self.b1 = bias([2048])
    self.bng5 = gamma([2048])
    self.bnb5 = beta([2048])
    self.bnm5 = beta([2048])
    self.bnv5 = beta([2048])

    # FC-2
    self.w2 = weight([2048,2048])
    self.b2 = bias([2048])
    self.bng6 = gamma([2048])
    self.bnb6 = beta([2048])
    self.bnm6 = beta([2048])
    self.bnv6 = beta([2048])

    # FC-3
    self.w3 = weight([2048,2])
    self.b3 = bias([2])
    self.conv_para = {'stride':1,
                     'pad':'SAME'}
    self.conv_para2 = {'stride':2,
                      'pad':'SAME'}
    #self.pool_para = {'stride':2,
    #                  'pad':'SAME#',
    #                  'kernel':2}

    self.regularizers = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    #########
    # Layer #
    #########
    self._cnn1 = cnn1d_relu_bn(self.x,self.f1,self.fb1,self.conv_para,self.bng1,self.bnb1,self.bnm1,self.bnv1)
    self.cnn1 = cnn1d_relu_bn(self._cnn1,self.half_f1,self.half_fb1,self.conv_para2,self.half_bng1,self.half_bnb1,self.half_bnm1,self.half_bnv1)
    #self.cnn1 = cnn1d_relu_bn(self.x,self.half_f1,self.half_fb1,self.conv_para2,self.half_bng1,self.half_bnb1)
    #print self.cnn1.get_shape()
    self._cnn2 = cnn1d_relu_bn(self.cnn1,self.f2,self.fb2,self.conv_para,self.bng2,self.bnb2,self.bnm2,self.bnv2)
    self.cnn2 = cnn1d_relu_bn(self._cnn2,self.half_f2,self.half_fb2,self.conv_para2,self.half_bng2,self.half_bnb2,self.half_bnm2,self.half_bnv2)
    #print self.cnn2.get_shape()
    self._cnn3 = cnn1d_relu_bn(self.cnn2,self.f3,self.fb3,self.conv_para,self.bng3,self.bnb3,self.bnm3,self.bnv3)
    self.cnn3 = cnn1d_relu_bn(self._cnn3,self.half_f3,self.half_fb3,self.conv_para2,self.half_bng3,self.half_bnb3,self.half_bnm3,self.half_bnv3)
    self.cnn4 = cnn1d_relu_bn(self.cnn3,self.f4,self.fb4,self.conv_para,self.bng4,self.bnb4,self.bnm4,self.bnv4)
    # flatten last cnn layer's output
    self.cnn4_output = tf.reshape(self.cnn4,[-1,256*13])
    #self.dnn1 = dnn_relu(self.cnn4_output,self.w1,self.b1)
    #self.dnn2 = dnn_relu(self.dnn1,self.w2,self.b2)
    self.dnn1 = dnn_relu_bn(self.cnn4_output,self.w1,self.b1,self.bng5,self.bnb5,self.bnm5,self.bnv5)
    self.dnn2 = dnn_relu_bn(self.dnn1,self.w2,self.b2,self.bng6,self.bnb6,self.bnm6,self.bnv6)
    # last layer without relu!!!!
    self.dnn3 = dnn(self.dnn2,self.w3,self.b3)
    # dnn3_1 silhouette & dnn3_2 distance
    #self._dnn3_1 = sigmoid(self._dnn3[:,:1024])
    #self._dnn3_2 = tf.reshape(self._dnn3[:,1024],[-1,1])
    #print self._dnn3_1.get_shape()
    #print self._dnn3_2.get_shape()
    #raw_input()
    #self.dnn3 = tf.concat(1,[self._dnn3_1,self._dnn3_2])
    #self.softmax = softmax(self.dnn2)

  def loss(self,X,y,X1,y1,e,mode='test',lr=2e-4,reg=1e-5,batch=5,epoch=21,opt=True,verbose=True):
    # history record 
    self.X_loss_history = []
    self.X1_loss_history = []
    self.X_acc_history = []
    self.X1_acc_history = []
    self.X_err_history = []
    self.X1_err_history = []

    # loss function
    #cross_entropy = -tf.reduce_sum(self.y_hat*tf.log(self.softmax))
    _reg = tf.constant(reg)
    err = tf.reduce_sum(tf.square(tf.sub(self.dnn3,self.y_hat)))
    cross_entropy = tf.add(err, tf.mul(_reg,self.regularizers))

    # optimizer
    #f = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    #f = tf.train.MomentumOptimizer(lr,0.9).minimize(cross_entropy)
    #f = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(cross_entropy)
    f = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)
    #f = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # outcome
    #pred = tf.equal(tf.argmax(self.softmax,1),tf.argmax(self.y_hat,1))
    #pred = tf.equal( tf.cast(tf.less(self._dnn3_1,0.5),tf.float32),
    #                 tf.cast(tf.less(self.y_hat[:,:1024],0.5),tf.float32) )
    #acc = tf.reduce_mean(tf.cast(pred,tf.float32))
    #err = tf.reduce_mean(tf.square(tf.sub(self.dnn3,self.y_hat)))

    # initialize session & saver
    if verbose: _log(1,e,'echo ini var and saver start>> tmp')
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    last_saver = tf.train.Saver()
    #sess = tf.InteractiveSession()
    sess = tf.Session()
    sess.run(init)
    if verbose:_log(0,e,'echo ini var and saver end>> tmp')

    #########
    # Train #
    #########
    if mode=='train':
      # whether restart new training session
      if opt:
        #good_record = 0.0
        low_loss = np.inf
        low_err = np.inf
      else:
        if verbose:_log(1,e,'echo load para start>> tmp')
        saver.restore(sess, "model.ckpt")

        X1_loss = 0
        #X1_acc = 0
        X1_err = 0

        for X1_idx in range(X1.shape[0]):
          _loss,_error = sess.run([cross_entropy,err],
                                      feed_dict={self.x:X1[X1_idx:X1_idx+1],
                                                 self.y_hat:y1[X1_idx:X1_idx+1]})
          X1_loss += _loss
          #X1_acc += _accuracy
          X1_err += _error

        low_loss = X1_loss
        #good_record = X1_acc/X1.shape[0]
        low_err = X1_err/X1.shape[0]
        if verbose:_log(0,e,'echo load para end>> tmp')

      num = X.shape[0]
      for i in range(epoch):
        for j in range(num/batch):

          if verbose:_log(1,e,'echo random minibatch start>> tmp')
          batch_xs, batch_ys = random_minibatch(X,y,batch)
          if verbose:_log(0,e,'echo random minibatch end>> tmp')

          if verbose:_log(1,e,'echo train start>> tmp')
          sess.run(f,feed_dict={self.x:batch_xs,self.y_hat:batch_ys})
          if verbose:_log(0,e,'echo train end>> tmp')

          # every 10 iter check status (in order to eliminate bn => feed it one by one)
          if j%100==0:
            if verbose:_log(1,e,'echo check all train data start>> tmp')

            X_loss = 0
            #X_acc = 0
            X_err = 0

            for X_idx in range(X.shape[0]):
              _loss,_error = sess.run([cross_entropy,err],
                                          feed_dict={self.x:X[X_idx:X_idx+1],
                                                     self.y_hat:y[X_idx:X_idx+1]})
              X_loss += _loss
              #X_acc += _accuracy
              X_err +=_error
            if verbose:_log(0,e,'echo check all train data end>> tmp')

            X1_loss = 0
            #X1_acc = 0
            X1_err = 0

            if verbose:_log(1,e,'echo check all val data start>> tmp')
            for X1_idx in range(X1.shape[0]):
              _loss1,_error1 = sess.run([cross_entropy,err],
                                            feed_dict={self.x:X1[X1_idx:X1_idx+1],
                                                       self.y_hat:y1[X1_idx:X1_idx+1]})
              X1_loss += _loss1
              #X1_acc += _accuracy1
              X1_err += _error1
            if verbose:_log(0,e,'echo check all val data end>> tmp')

            loss = X_loss
            loss1 = X1_loss
            #accuracy = X_acc/X.shape[0]
            #accuracy1 = X1_acc/X1.shape[0]
            error = X_err/X.shape[0]
            error1 = X1_err/X1.shape[0]

            self.X_loss_history += [loss]
            self.X1_loss_history += [loss1]
            #self.X_acc_history += [accuracy]
            #self.X1_acc_history += [accuracy1]
            self.X_err_history += [error]
            self.X1_err_history += [error1]

            # save best record
            #if accuracy1 >= good_record and error1 < low_err:
            if error1 < low_err:
              #good_record = accuracy1
              low_loss = loss1
              low_err = error1

              if verbose:_log(1,e,'echo save para start>> tmp')
              save_path = saver.save(sess, "model.ckpt",write_meta_graph=False)
              if verbose:_log(0,e,'echo save para end>> tmp')

              print("######## Model saved in file: %s ########" % save_path)

            print("epoch %2d/%2d,\titer %2d/%2d," %(i,epoch,j,num/batch))
            print("Train Loss %.10f  Error %.10f" %(loss,error))
            print("Valid Loss %.10f  Error %.10f" %(loss1,error1))

      # Force to save the last parameters
      save_path = last_saver.save(sess, "last_model.ckpt",write_meta_graph=False)

    ########
    # Test #
    ########
    elif mode=='test':
      saver.restore(sess, "last_model.ckpt")
      print("Model restored.")
      loss,accuracy,error,org_output = sess.run([cross_entropy,acc,err,self.dnn3],feed_dict={self.x:X1,self.y_hat:y1})
      print("Vali Loss %.10f Accuracy %.10f Error %.10f" %(loss,accuracy,error))
      loss1,accuracy1,error1,org_output1 = sess.run([cross_entropy,acc,err,self.dnn3],feed_dict={self.x:X,self.y_hat:y})
      print("Test Loss %.10f Accuracy %.10f Error %.10f" %(loss1,accuracy1,error1))
      return org_output, org_output1

    #sess.close()
    #del sess
    #del saver

def test():
  pass

  e = mp.Event()

  X = np.random.normal(0,1,(10,1,2048,13))
  y = np.random.randint(2,size=(10,2)).astype(np.int32)
  X1 = np.random.normal(0,1,(10,1,2048,13))
  y1 = np.random.randint(2,size=(10,2)).astype(np.int32)


  net = modelX()
  net.loss(X,y,X1,y1,e,mode='train',lr=1e-2,reg=0.0,batch=10,epoch=100)
  #net.loss(X,y,X1,y1,e,mode='train',lr=1e-1,reg=1e-2,batch=10,epoch=100)

  #print max(net.X1_acc_history)
  print min(net.X1_loss_history)
  print min(net.X1_err_history)

if __name__=='__main__':
  test()
  pass
