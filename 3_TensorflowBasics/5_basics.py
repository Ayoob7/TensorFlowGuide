import tensorflow as tf
import numpy as np

#20X + 10
weights = tf.constant(20)
inputs = tf.placeholder(dtype=tf.int32)
bias = tf.constant(10)

inputSum = tf.add(tf.multiply(weights,inputs),bias)

with tf.Session() as sess:
    print(sess.run(inputSum,feed_dict={inputs:np.arange(0,100,1)}))


#matmul
# a = tf.constant([[1,2,1],[5,6,4],[2,1,1]])
# b = tf.constant([[20],[10],[20]])
# op1 = tf.matmul(a,b)
#
# with tf.Session() as sess:
#     print(sess.run(op1))