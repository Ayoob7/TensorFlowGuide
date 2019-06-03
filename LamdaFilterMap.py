import numpy as np
from functools import reduce

meter = np.random.randint(0,100,20)
print("Input : ",list(meter))

# map
cm = list(map((lambda m: m*100),meter))
print("Map : ",cm)

# filter
filteredCM = list(filter((lambda cm:cm > 3500),cm))
print("Filter : ",filteredCM)

# lambda
prodPlus4 = lambda x, y: x * y + 4
print("Lambda : ", prodPlus4(10, 10))

# reduce
n = [1,2,3,4]
addAllFilteredCM = reduce((lambda x,y:x+y),n)
print("Reduce : ",addAllFilteredCM)