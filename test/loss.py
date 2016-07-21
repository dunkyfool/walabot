import numpy as np
import tensorflow as tf
from time import time

x = tf.placeholder(tf.float32,[None,2])
y = tf.placeholder(tf.float32,[None,2])

w1 = tf.Variable(tf.random_uniform([5,2], minval=-1e-1, maxval=1e-1, dtype=tf.float32),name='w1')
b1 = tf.Variable(tf.random_uniform([2], minval=-1e-1, maxval=1e-1, dtype=tf.float32),name='b1')

#f1 = tf.matmul(x,w1)
#f2 = tf.add(f1,b1)

loss = tf.reduce_sum(tf.square(tf.sub(x,y)))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

data = np.array([-0.5,0.5]).reshape(1,2)
label = np.array([1,0]).reshape(1,2)

print sess.run(loss,feed_dict={x:data,y:label})
