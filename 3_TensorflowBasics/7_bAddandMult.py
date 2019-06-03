import tensorflow as tf
import numpy as np

np.random.seed(101)
tf.set_random_seed(101)

randA = np.random.uniform(0,100,(5,5))
print(randA)

randB = np.random.uniform(0,100,(5,1))
print(randB)


plcA = tf.placeholder(tf.float32)
plcB = tf.placeholder(tf.float32)

opAdd = plcA + plcB
opMul = plcA * plcB

with tf.Session() as sess:
    print(sess.run(opAdd,feed_dict={plcA:randA,plcB:randB}))