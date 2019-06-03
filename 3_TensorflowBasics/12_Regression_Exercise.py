import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Data ingress
housingData = pd.read_csv('data/cal_housing_clean.csv')

# Feature and labels
features = housingData.drop("medianHouseValue",axis=1)
labels = housingData['medianHouseValue']

# Test Train split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# Scaler and to Normalize
scaler = MinMaxScaler(copy=True , feature_range=(0,1) )

# Scaled version of features
X_train = pd.DataFrame(data=scaler.fit_transform(X_train),
                       columns=X_train.columns,
                       index=X_train.index)

X_test = pd.DataFrame(data=scaler.fit_transform(X_test),
                       columns=X_test.columns,
                       index=X_test.index)

# Feature Columns
housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
totalRooms = tf.feature_column.numeric_column('totalRooms')
totalBedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [housingMedianAge,totalRooms,totalBedrooms,population,households,medianIncome]

# Model
model = tf.estimator.LinearRegressor(feature_columns=feat_cols,label_dimension=1)

# Train
inputFunc = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,num_epochs=1000,shuffle=True)

model.train(input_fn=inputFunc,steps=100)

# Evaluate
inputFuncEval = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=20,num_epochs=1,shuffle=False)

results = model.evaluate(input_fn=inputFuncEval)

print(results)

predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
pred_gen = model.predict(predict_input_func)

predictions = list(pred_gen)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

print(mean_squared_error(y_test,final_preds)**0.5)


# # Model
# model = tf.estimator.DNNRegressor(hidden_units=[10,3,2],feature_columns=feat_cols,label_dimension=1)
#
# # Train
# inputFunc = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,num_epochs=1000,shuffle=True)
# model.train(input_fn=inputFunc,steps=1000)
#
# # Eval
# inputFuncEval = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=20,num_epochs=1,shuffle=False)
# results = model.evaluate(input_fn=inputFuncEval)
# print(results)
#
# # Predict
# inputFuncPred = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=20, num_epochs=1,shuffle=True)
# predictions = model.predict(input_fn=inputFuncPred)

finalPred = []
for pred in list(predictions):
    finalPred.append(pred['predictions'])

print(mean_squared_error(y_test,finalPred)**2)