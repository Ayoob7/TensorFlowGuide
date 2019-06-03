import numpy as np
import matplotlib.pyplot as plt

#mat = np.arange(0,10000)
mat = np.random.randint(0,10000,(100,100))
print(mat)
plt.imshow(mat,cmap='coolwarm')
plt.ylabel("Temporal Distribution")
plt.xlabel("Events Distribution")
plt.show()