import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

#singleImage = mnist.train.images.reshape(28,28)
# plt.imshow(singleImage)
# plt.show()

# PlaceHolders
x = tf.placeholder(tf.float32,shape=[None,784])

# Variables
w1 = tf.Variable(tf.random_normal(shape=[784,10]))
b1 = tf.Variable(tf.random_normal(shape=[10]))

# Graph operations
xXw1 = tf.matmul(x,w1)
y = tf.add(xXw1,b1)


# Loss Function
yTrue = tf.placeholder(tf.float32,shape=[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yTrue,logits=y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Session start
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Train
    for step in range(1000):
        batchX, batchY = mnist.train.next_batch(100)

        sess.run(train,feed_dict={x:batchX,yTrue:batchY})

    # Evaluation
    correctPred = tf.equal(tf.argmax(y,1) , tf.arg_max(yTrue,1))

    acc = tf.reduce_mean( tf.cast(correctPred,tf.float32) )

    print(sess.run(acc,feed_dict={x:mnist.test.images,yTrue:mnist.test.labels}))
