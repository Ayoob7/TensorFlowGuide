# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Environment Settings
pd.set_option('display.expand_frame_repr', False)

# Data Ingress
censusData = pd.read_csv('data/census_data.csv')

# Feature Engineering
censusData['income_bracket'] = censusData['income_bracket'].apply(lambda label: int(label == ' <=50K'))

# Data pre processing
features = censusData.drop('income_bracket',axis=1)
labels = censusData['income_bracket']

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Feature Columns (Catergorical) if you use DNN Classifier then have to convert the catergorical to embedded ones (as in 11_Classifier)
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Feature Columns (Numerical)
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# Feature Columns List
feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]


#### Model
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

# Training input function
inputFunc = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=100,shuffle=True)

model.train(input_fn=inputFunc,steps=1)


# Evaluation
inputFuncEval = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=100,num_epochs=None,shuffle=False)

results = model.evaluate(inputFuncEval)
print(results)

# Predictions
inputFuncPred = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),num_epochs=None,shuffle=False)

predictions = model.predict(input_fn=inputFuncPred)

predictionList = list(predictions)

for i in predictionList:
    print(i)
