import numpy as np
import tensorflow as tf
from utils.layers import *
from utils.initParams import *
from utils.monitor2 import *
# All DNN w/o softmax
# DNN-262144 DNN-262144 DNN-262144 DNN-2
#

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
    self.x = tf.placeholder(tf.float32,[None,327680])
    self.y_hat = tf.placeholder(tf.float32,[None, 2])
    #self.f1 = weight([3,3,3,3])
    #self.fb1 = bias([3])
    #self.bng1 = gamma([3])
    #self.bnb1 = beta([3])
    #self.half_f1 = weight([4,4,3,3])
    #self.half_fb1 = bias([3])
    #self.half_bng1 = gamma([3])
    #self.half_bnb1 = beta([3])
    #self.f2 = weight([3,3,3,3])
    #self.fb2 = bias([3])
    #self.bng2 = gamma([3])
    #self.bnb2 = beta([3])
    #self.half_f2 = weight([4,4,3,3])
    #self.half_fb2 = bias([3])
    #self.half_bng2 = gamma([3])
    #self.half_bnb2 = beta([3])
    #self.f3 = weight([3,3,3,3])
    #self.fb3 = bias([3])
    #self.bng3 = gamma([3])
    #self.bnb3 = beta([3])
    #self.f4 = weight([3,3,3,3])
    #self.fb4 = bias([3])
    #self.bng4 = gamma([3])
    #self.bnb4 = beta([3])
    self.w1 = weight([327680,500])
    self.b1 = bias([500])
    self.w2 = weight([500,100])
    self.b2 = bias([100])
    self.w3 = weight([100,50])
    self.b3 = bias([50])
    self.w4 = weight([50,2])
    self.b4 = bias([2])
    #self.conv_para = {'stride':1,
    #                 'pad':'SAME'}
    #self.conv_para2 = {'stride':2,
    #                  'pad':'SAME'}
    #self.pool_para = {'stride':2,
    #                  'pad':'SAME#',
    #                  'kernel':2}

    self.regularizers = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    #########
    # Layer #
    #########
    #self._cnn1 = cnn_relu_bn(self.x,self.f1,self.fb1,self.conv_para,self.bng1,self.bnb1)
    #self.cnn1 = cnn_relu_bn(self._cnn1,self.half_f1,self.half_fb1,self.conv_para2,self.half_bng1,self.half_bnb1)
    #print self.cnn1.get_shape()
    #self._cnn2 = cnn_relu_bn(self.cnn1,self.f2,self.fb2,self.conv_para,self.bng2,self.bnb2)
    #self.cnn2 = cnn_relu_bn(self._cnn2,self.half_f2,self.half_fb2,self.conv_para2,self.half_bng2,self.half_bnb2)
    #print self.cnn2.get_shape()
    #self.cnn3 = cnn_relu_bn(self.cnn2,self.f3,self.fb3,self.conv_para,self.bng3,self.bnb3)
    #self.cnn4 = cnn_relu_bn(self.cnn3,self.f4,self.fb4,self.conv_para,self.bng4,self.bnb4)
    # flatten last cnn layer's output
    #self.cnn4_output = tf.reshape(self.cnn4,[-1,64*3])
    self.dnn1 = dnn_relu(self.x,self.w1,self.b1)
    self.dnn2 = dnn_relu(self.dnn1,self.w2,self.b2)
    self.dnn3 = dnn_relu(self.dnn2,self.w3,self.b3)
    self.dnn4 = dnn(self.dnn3, self.w4, self.b4)
    #self.softmax = softmax(self.dnn2)

  def loss(self,X,y,X1,y1,e,mode='test',lr=2e-4,reg=1e-5,batch=5,epoch=21,opt=True,verbose=True):
    # monitor record
    FILE='tmp'

    # history record 
    self.X_loss_history = []
    self.X1_loss_history = []
    self.X_acc_history = []
    self.X1_acc_history = []

    # loss function
    #cross_entropy = -tf.reduce_sum(self.y_hat*tf.log(self.softmax))
    cross_entropy = tf.reduce_sum(tf.square(tf.sub(self.dnn4,self.y_hat)))
    cross_entropy += reg*self.regularizers

    # optimizer
    #f = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    #f = tf.train.MomentumOptimizer(lr,0.9).minimize(cross_entropy)
    f = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(cross_entropy)

    # initialize session & saver
    if verbose:
      cmd = 'echo ini var and saver start>> tmp'
      os.system(cmd)
      e.set()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    #sess = tf.InteractiveSession()
    sess = tf.Session()
    sess.run(init)
    if verbose:
      cmd = 'echo ini var and saver end>> tmp'
      os.system(cmd)
      e.clear()

    # outcome
    #pred = tf.equal(tf.argmax(self.softmax,1),tf.argmax(self.y_hat,1))
    #acc = tf.reduce_mean(tf.cast(pred,tf.float32))

    #########
    # Train #
    #########
    if mode=='train':
      # whether restart new training session
      if opt:
        good_record = 0.0
        low_loss = np.inf
      else:
        if verbose:
          cmd = 'echo load para start>> tmp'
          os.system(cmd)
          e.set()
        saver.restore(sess, "model.ckpt")
        low_loss = sess.run(cross_entropy,feed_dict={self.x:X1,self.y_hat:y1})
        if verbose:
          cmd = 'echo load para end>> tmp'
          os.system(cmd)
          e.clear()

      num = X.shape[0]
      for i in range(epoch):
        for j in range(num/batch):
          if verbose:
            cmd = 'echo random minibatch start>> tmp'
            os.system(cmd)
            e.set()
          batch_xs, batch_ys = random_minibatch(X,y,batch)
          if verbose:
            cmd = 'echo random minibatch end>> tmp'
            os.system(cmd)
            e.clear()

          if verbose:
            cmd = 'echo train start>> tmp'
            os.system(cmd)
            e.set()
          sess.run(f,feed_dict={self.x:batch_xs,self.y_hat:batch_ys})
          if verbose:
            cmd = 'echo train end>> tmp'
            os.system(cmd)
            e.clear()
          # every 10 iter check status
          if j%100==0:
            if verbose:
              cmd = 'echo check all train data start>> tmp'
              os.system(cmd)
              e.set()
            loss = sess.run(cross_entropy,feed_dict={self.x:X,self.y_hat:y})
            if verbose:
              cmd = 'echo check all train data end>> tmp'
              os.system(cmd)
              e.clear()
            if verbose:
              cmd = 'echo check all val data start>> tmp'
              os.system(cmd)
              e.set()
            loss1 = sess.run(cross_entropy,feed_dict={self.x:X1,self.y_hat:y1})
            if verbose:
              cmd = 'echo check all val data end>> tmp'
              os.system(cmd)
              e.clear()
            self.X_loss_history += [loss]
            self.X1_loss_history += [loss1]
            #self.X_acc_history += [accuracy]
            #self.X1_acc_history += [accuracy1]
            # save best record
            #if accuracy1 >= good_record:# and loss1 < low_loss:
            if loss1 < low_loss:
              #good_record = accuracy1
              low_loss = loss1
              if verbose:
                cmd = 'echo save para start>> tmp'
                os.system(cmd)
                e.set()
              save_path = saver.save(sess, "model.ckpt",write_meta_graph=False)
              if verbose:
                cmd = 'echo save para end>> tmp'
                os.system(cmd)
                e.clear()
              print("!!Model saved in file: %s" % save_path)
            print("epoch %2d/%2d,\titer %2d/%2d," %(i,epoch,j,num/batch))
            print("Train Loss %.10f" %(loss))
            print("Valid Loss %.10f" %(loss1))
            if loss1 == np.nan:
              print 'nan issue'
              break
    ########
    # Test #
    ########
    elif mode=='test':
      saver.restore(sess, "model.ckpt")
      print("Model restored.")
      loss = sess.run(cross_entropy,feed_dict={self.x:X1,self.y_hat:y1})
      print("Vali Loss %.10f"%(loss))
      loss = sess.run(cross_entropy,acc,feed_dict={self.x:X,self.y_hat:y})
      print("Test Loss %.10f"%(loss))
      pass

    #sess.close()
    #del sess
    #del saver

if __name__=='__main__':
  pass
