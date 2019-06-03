from numpy import random,array,exp, dot

class NeuralNetwork():
    def __init__(self):
        #seed a random number generator , so it generates the same numbers everytime the program runs
     random.seed(1)
        #model a single neuron that has 3 inputs and 1 output
        #we assign random weights to a 3x1 matrix from values -1 to 1 and mean 0
     self.synaptic_weights = 2 * random.random((3,1)) - 1

        #pass the weighted sum of the inputs*weights into the Sigmoid Function to normalise them between 0 and 1
    def __sigmoid(self,x):
        return 1/(1+exp(-x))

    #the dy/dx of the Sigmoid Curve , we use this in the 'adjusted weights by' equation
    def __sigmoid_derivative(self,x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #pass the training set through our neural network (just one neuron so far)
            output=self.think(training_set_inputs)

            #calculate the error
            error = training_set_outputs - output

            #Calculate the adjustment less confident weights need by : input * error * Gradient of S curve
            adjustments = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #make the adjustments to the weights
            self.synaptic_weights += adjustments

    #the neural network thinks
    def think(self,inputs):
        #pass the inputs through our neural network to generate an output
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__=="__main__":
        neural_network = NeuralNetwork()

        print ("Random Starting Synaptic Weights:")
        print (neural_network.synaptic_weights)
        training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
        training_set_outputs = array([[0,1,1,0]]).T

        #training the neural network using a training set
        #do it 10,000 times and make small adjustments each time
        neural_network.train(training_set_inputs,training_set_outputs,100)

        print ("New synaptic weights after training")
        print (neural_network.synaptic_weights)

        print ("Considering new situation [1,0,0] -> ?: ")
        print (neural_network.think(array([1,0,0])))