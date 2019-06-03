import numpy as np

class Graph():
    def __init__(self):
        global singleGraph
        self.listOfVariables = []
        self.listOfPlaceholders = []
        self.listOfOperations = []
        singleGraph = self


class Variable():
    def __init__(self,initVal=None):
        self.value = initVal
        self.outputNodes = []

        singleGraph.listOfVariables.append(self)


class Placeholder():
    def __init__(self):
        self.outputNodes = []

        singleGraph.listOfPlaceholders.append(self)


class Operator():
    def __init__(self,inputNodes = []):
        self.inputNodes = inputNodes
        self.outputNodes = []

        for node in inputNodes:
            node.outputNodes.append(self)

        singleGraph.listOfOperations.append(self)

    def compute(self):
        pass
# Operators
class add(Operator):
    def __init__(self,x,y):
        super().__init__([x,y])

    def compute(self,x,y):
        return x+y

class multiply(Operator):
    def __init__(self,x,y):
        super().__init__([x,y])
    def compute(self,x,y):
        return x*y

class square(Operator):
    def __init__(self,x):
        super().__init__([x])
    def compute(self,x):
        return x*x

class matmul(Operator):
    def __init__(self,x,y):
        super().__init__([x,y])
    def compute(self,x,y):
        return x.dot(y)

class sigmoid(Operator):
    def __init__(self,x):
        super().__init__([x])
    def compute(self,x):
        return 1/ (1+np.exp(-x))
# End of Operators

def makeNodesInOrder(operation):
    nodesInOrder = []
    def recurse(node):
        if isinstance(node, Operator):
            for inputNodes in node.inputNodes:
                recurse(inputNodes)
        nodesInOrder.append(node)
    recurse(operation)
    return nodesInOrder


class Session:

    def run(self, operation, feedDict={}):

        nodesInorder = makeNodesInOrder(operation)

        for node in nodesInorder:
            if type(node) == Placeholder:
                node.output = feedDict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                node.inputs = [inputNodes.output for inputNodes in node.inputNodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output

graph = Graph()
#10X +1
# var1 = Variable(10)
# var2 = Variable(1)
# plc1 = Placeholder()
#
# op1 = multiply(var1,plc1)
# op2 = add(var2,op1)


#x^2 + 2x +1
# var1 = Variable(2)
# var2 = Variable(1)
# plc1 = Placeholder()
#
# op1 = square(plc1)
# op2 = multiply(var1,plc1)
# op3 = add(op1,op2)
# op4 = add(op3,var2)

#matrix multiplication
# var1 = Variable([[10, 20], [30, 40]])
# var2 = Variable([1,1])
# plc1 = Placeholder()
# op1 = matmul(var1, plc1)
# op2 = add(op1,var2)

# sess = Session()
# result = sess.run(operation=op4, feedDict={plc1:11})
# print(result)


from sklearn.datasets import make_blobs

data = make_blobs(n_samples=50,n_features=2,centers=2,random_state=75)

features = data[0]
labels = data[1]


import matplotlib.pyplot as plt
x = np.linspace(0,11,10)
y = -x + 5
plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')
plt.scatter(x,y)
plt.show()

xInput = Placeholder()
weights = Variable([1,1])
bias = Variable(-5)

summat = add(matmul(xInput,weights),bias)

normalisedSummat = sigmoid(summat)

sess = Session()

res = sess.run(normalisedSummat,feedDict={xInput:[2,-10]})
print(res)
