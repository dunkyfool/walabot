import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[None,5])
y_hat = tf.placeholder(tf.float32,[None,5])
w = tf.Variable(tf.ones([5,5]))

f1 = tf.matmul(x,w)
f2 = tf.add(y_hat,1)

c1 = tf.less(x,0.5)
c2 = tf.less(y_hat,0.5)
#c3 = tf.logical_xor(c1,c2)

d1 = tf.cast(c1,tf.float32)
d2 = tf.cast(c2,tf.float32)

pred = tf.equal(d1,d2)
acc = tf.reduce_mean(tf.cast(pred,tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

data = np.array([0,1,0,1,0]).reshape(1,5)
label = np.array([0,1,0,1,1]).reshape(1,5)


a,b,c,d,e,f = sess.run([c1,c2,d1,d2,pred,acc],feed_dict={x:data,y_hat:label})
print 'x_mask';print a
print 'y_mask';print b
print 'x_mask_cast';print c
print 'y_mask_cast';print d
print 'pred';print e
print 'acc';print f
