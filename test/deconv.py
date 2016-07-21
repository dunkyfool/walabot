import sys
import tensorflow as tf
import numpy as np

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
#print sys.argv[1]

#def dot(sess,x,w):
#  update_w = tf.assign(w,w*0.5)
#  with tf.control_dependencies([update_w]):
##    sess.run(w.assign(w*0.5))
#    return tf.clip_by_value(tf.matmul(x,w),0.3,1)


if __name__=='__main__':
    x = tf.placeholder(tf.float32,[None,4,4,1])
    y = tf.placeholder(tf.float32,[None,1])
    sess = tf.Session()

    w_value = np.array([1,1,1,1,1,1,1,1,1]).reshape((3,3,1,1))
    w = tf.Variable(w_value,dtype=tf.float32)
    z1 = tf.nn.conv2d_transpose(x,w,output_shape=[1,4,4,1],strides=[1,1,1,1],padding='SAME')
    z2 = tf.nn.conv2d(z1,w,strides=[1,1,1,1],padding='SAME')

    data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(1,4,4,1)
    label = np.array([1]).reshape(1,1)

    init = tf.initialize_all_variables()

    sess.run(init)

    o1,o2 = sess.run([z1,z2],feed_dict={x:data})
    print 'z1',o1
    print 'z1',o1.shape
    print 'z2',o2
    print 'z2',o2.shape


