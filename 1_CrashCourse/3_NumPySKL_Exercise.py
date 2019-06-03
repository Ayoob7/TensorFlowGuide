import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(101)

mat = np.random.randint(0, 1000, (100, 100))
# plt.imshow(mat)
# plt.show()

df = pd.DataFrame(mat)
df.plot(x=0,y=1,kind="scatter")
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(mat)
scaledDf = scaler.transform(mat)

# plt.imshow(scaledDf)
# plt.show()

colList = ["f"+str(i) for i in range(1,100)]
colList.append("Label")

dfFinal = pd.DataFrame(data=scaledDf,columns=colList)

features = dfFinal[colList[0:len(colList)-1]]
labels = dfFinal[[colList.pop()]]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

