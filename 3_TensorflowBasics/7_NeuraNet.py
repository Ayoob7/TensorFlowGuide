import tensorflow as tf
import numpy as np

noFeatures = 10
noNeurons = 3

# before putting the data please fill its shape properly

inputs = tf.placeholder(tf.float32,(None,noFeatures))
weights = tf.Variable(tf.random_normal([noFeatures,noNeurons]))
bias = tf.Variable(tf.ones([noNeurons]))

inputXweights = tf.matmul(inputs,weights)
plusBias = tf.add(inputXweights,bias)
activation = tf.sigmoid(plusBias)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(activation,feed_dict={inputs: np.random.random([1,noFeatures])}))