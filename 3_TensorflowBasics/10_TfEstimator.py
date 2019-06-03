import tensorflow as tf
import numpy as np

xData = np.linspace(0,1,1000000)
noise = np.random.randn(len(xData))

yTrue = 0.5*xData + 5 + noise


feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(xData,yTrue,test_size=0.3, random_state = 101)

input_func = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)

eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False)



estimator.train(input_fn=input_func,steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)

eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)


print("train metrics: {}".format(train_metrics))
print("eval metrics: {}".format(eval_metrics))


input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)
list(estimator.predict(input_fn=input_fn_predict))


predictions = []# np.array([])
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])
    print(x['predictions'])