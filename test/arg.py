import sys
import tensorflow as tf
import numpy as np

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
#print sys.argv[1]

def dot(sess,x,w):
  update_w = tf.assign(w,w*0.5)
  with tf.control_dependencies([update_w]):
#    sess.run(w.assign(w*0.5))
    return tf.clip_by_value(tf.matmul(x,w),0.3,1)


if __name__=='__main__':
    x = tf.placeholder(tf.float32,[None,3])
    y = tf.placeholder(tf.float32,[None,3])
    sess = tf.Session()

    w_value = np.array([1,0,0]).reshape((3,1))
    w = tf.Variable(w_value,dtype=tf.float32)
    z = dot(sess,x,w)

    data = np.array([1.0,-1,0.0,2.0,1.0,0.3]).reshape(2,3)
    label = np.array([0,0,1,0,0,1]).reshape(2,3)

    init = tf.initialize_all_variables()

    sess.run(init)

    o1 = sess.run([z],feed_dict={x:data})
    print 'z',o1
    o1 = sess.run([w],feed_dict={x:data})
    print 'w',o1

    #sess.run(w.assign(w*0.5))

    o1 = sess.run([z],feed_dict={x:data})
    print 'z',o1
    o1 = sess.run([w],feed_dict={x:data})
    print 'w',o1

