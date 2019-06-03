import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data  Ingress
diabetes = pd.read_csv('data/pima-indians-diabetes.csv')

# Cols to normalize
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min() ))

# Individual Columns (Numeric)
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')


# Individual Columns (Catergorical)
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

# Feature Engineering (Age)
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])

# Feature Columns
feature_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_bucket]

# Test Train Split (Features)
x_data = diabetes.drop('Class',axis=1)

# Test Train Split (Labels)
labels = diabetes['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)


################################################
# Model (Finally !!!) - Just a linear classifier
model = tf.estimator.LinearClassifier(feature_columns=feature_cols,n_classes=2)

# Train
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model.train(input_fn=input_func,steps=1000)

# Evaluation
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)

results = model.evaluate(eval_input_func)

print(results)


# Predictions
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=True)

predictions = model.predict(pred_input_func)

################################################
print("Dense Neural Network")
embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4)

feature_cols_DNN = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,embedded_group_col,age_bucket]

# Model (Finally !!!) - Dense Neural Network
modelDNN = tf.estimator.DNNClassifier(hidden_units= [10],feature_columns=feature_cols_DNN,n_classes=2)

# Train
input_funcDNN = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

modelDNN.train(input_funcDNN)# Will get an error because we need a emberdding space for categorical column.


# Evaluation
eval_input_func_DNN = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)

resultsDNN = modelDNN.evaluate(eval_input_func_DNN)
print(resultsDNN)

# Predictions
pred_input_func_DNN = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=True)

predictionsDNN = modelDNN.predict(pred_input_func_DNN)

