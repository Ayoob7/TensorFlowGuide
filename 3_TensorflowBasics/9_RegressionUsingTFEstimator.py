import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

xData = np.linspace(0,1,1000000)
noise = np.random.randn(len(xData))

yTrue = 0.5*xData + 5 + noise

xDf = pd.DataFrame(data=xData,columns=['X Data'])
yDf = pd.DataFrame(data=yTrue,columns=['Y'])

myDf = pd.concat([xDf,yDf],axis=1)

myDf.sample(n=250).plot(kind="scatter",x="X Data",y="Y")
plt.show()

#Data

batchSize = 8

rand2 = np.random.randn(2)

m = tf.Variable(rand2[0])
b = tf.Variable(rand2[1])

xPlc = tf.placeholder(tf.float64,[batchSize])
yPlc = tf.placeholder(tf.float64,[batchSize])


# Equation
yModel = m*xPlc + b

error = tf.reduce_sum(tf.square(yPlc - yModel))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    batches = 1000

    for i in range(batches):
        randInd = np.random.randint(len(xData),size=batchSize)

        feed = {xPlc:xData[randInd],yPlc:yTrue[randInd]}

        sess.run(train,feed_dict=feed)

    modelM, modelB = sess.run([m,b])


print(modelM,modelB)