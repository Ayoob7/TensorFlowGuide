import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

xData = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
yData = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

m = tf.Variable(np.random.rand(2)[0])
b = tf.Variable(np.random.rand(2)[1])

error = tf.Variable(0)

for x,y in zip(xData,yData):
    yPred = m*x +b

    #Cost Function
    error += (y-yPred)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    trainingSteps = 2
    for i in range(trainingSteps):
        sess.run(train)
    finalM , finalB = sess.run([m,b])

xTestData = np.linspace(-1,11,10)
print(finalM,finalB)
yPredData = finalM*xTestData + finalB

plt.plot(xTestData,yPredData,"r")
plt.plot(xData,yData,"*")
plt.show()