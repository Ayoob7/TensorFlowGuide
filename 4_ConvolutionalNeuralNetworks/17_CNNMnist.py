import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# singleImage = mnist.test.images[2].reshape(28,28)
# plt.imshow(singleImage)
# plt.show()

# Helper Functions
# Init Weight
def init_weights(shape):
    initRandDist = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initRandDist)

# Init Bias
def init_bias(shape):
    initBiasVal = tf.constant(0.1,shape=shape)
    return tf.Variable(initBiasVal)

# Conv2D
def conv2d(x,W):
    # x --> [batch,H,W,Channels]
    # W --> [filter H, filter W, Channels IN, Channels OUT]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# Pooling
def max_pool_2by2(x):
    # x --> [batch,H,W,Channels]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# End of Helper Functions


## Create Conv Layer
def convolutional_layer(inputX,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(inputX,W)+b)

# Normal (fully connected)
def normal_full_layer(input_layer,size):
    input_size = int(input_layer.getshape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.add(tf.matmul(input_layer,W),b)