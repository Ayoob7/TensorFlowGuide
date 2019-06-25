import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sys
import re

#### PROGRESS BAR
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)

######

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
    steps = 1000
    progress = ProgressBar(steps, fmt=ProgressBar.FULL)
    # Train
    for step in range(1000):
        progress.current += 1
        progress()
        batchX, batchY = mnist.train.next_batch(100)

        sess.run(train,feed_dict={x:batchX,yTrue:batchY})
    progress.done()
    # Evaluation
    correctPred = tf.equal(tf.argmax(y,1) , tf.arg_max(yTrue,1))

    acc = tf.reduce_mean( tf.cast(correctPred,tf.float32) )

    print(sess.run(acc,feed_dict={x:mnist.test.images,yTrue:mnist.test.labels}))
