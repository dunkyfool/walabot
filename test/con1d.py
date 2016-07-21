import tensorflow as tf
import numpy as np

f_value = np.array([1,2,3,4]).reshape((1,4,1,1))

x = tf.placeholder(tf.float32,[None,1,6,1])
#f = tf.Variable(tf.zeros([1,3,3,3]))
f = tf.Variable(f_value,dtype=tf.float32)
b = tf.Variable(tf.constant(0.1,shape=[1]))

_y = tf.nn.conv2d(x,f,strides=[1,1,2,1],padding='SAME')
y = tf.add(_y,b)

para = []
para = [x] + [f] + [b] + [_y] + [y]
for item in para:
  print item.get_shape().as_list()


data = np.ones((2,1,6,1))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print sess.run(y,feed_dict={x:data})
