import tensorflow as tf

myTensor = tf.random_uniform((4,4),0,1)
myVar = tf.Variable(myTensor)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Note you can directly run a script with tensor like this but cant run a Variable without initializing globally first
    # print(sess.run(myTensor)) is fine BUT print(sess.run(myVar)) is not, cz its a variable
    sess.run(init)
    print(sess.run(myVar))

