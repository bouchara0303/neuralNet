import numpy as np
import matplotlib.pyplot as plt
from random import choice
import matplotlib as mpl

class Layer:
    """
    node - an int specifying the size of the layer
    """
    def __init__(self, node = 1):
        self.inputs = np.random.random_sample(node)
        self.outputs = np.zeros(node)

class NeuralNet:
    """
    layers - a list of ints specifying the shape of the layer at the corresponding index
    """
    def __init__(self, layers):

        #Ensure layers is an array
        assert(type(layers) == type([]))

        #Layers of neural network
        self.layers = [Layer(x) for (x) in layers]
        self.weights = [np.array([0])]
        self.bias = [np.array([0])]

        #Initialize random values for weights -- [0, 1) --  and zeros for biases
        for i in range(len(layers) - 1):
            self.weights.append(np.random.random_sample((layers[i + 1], layers[i])))
            self.bias.append(np.zeros((self.weights[i + 1].shape[0], 1)))

    #Propagate input values forward by multiplying by weights and adding bias
    def forwardProp(self, xTrain):
        self.layers[0].outputs = xTrain
        for i in range(len(self.layers) - 1):
            i += 1
            prevLayer = self.layers[i - 1]
            layer = self.layers[i]
            layer.inputs = NeuralNet.weightedSum(self.weights[i], prevLayer.outputs, self.bias[i])
            layer.outputs = NeuralNet.sigmoid(layer.inputs)

    #Propagate error backwards to update weights
    def backProp(self, xTrain, yTrain, learnRate):

        #Get inputs and outputs for layers
        self.forwardProp(xTrain)

        #Derivative of cost function w.r.t. weighted input for each node
        errorSignal = []
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i + 1]
            m,n,o = i, i + 1, i + 2

            #Get error signal for last layer to propagate backwards
            if i == (len(self.layers) - 2):
                errorSignal.insert(0, NeuralNet.costDerWeightedSum(layer.outputs, yTrain, layer.inputs))

            else:
                weightsNext = self.weights[o]
                errorSignal.insert(0, NeuralNet.errorSignalHidden(weightsNext, errorSignal[0], layer.inputs))

            #Derivatives used to update bias and weights
            prevLayerOut = self.layers[m].outputs
            deltaBias = errorSignal[0]
            deltaWeights = NeuralNet.costDerWeights(errorSignal[0], prevLayerOut)

            #Update weights and biases for each layer
            self.weights[n] = self.weights[n] - (learnRate * deltaWeights)
            self.bias[n] = self.bias[n] - (learnRate * deltaBias)

    #Train network
    def train(self, xTrain, yTrain, epochs, learnRate):
        for epoch in range(epochs):
            for i in range(len(xTrain)):
                i = choice(range(len(xTrain)))
                input, output = np.array(xTrain[i]), np.array(yTrain[i])
                self.backProp(input, output, learnRate)

    #Predict output
    def predict(self, input):
        self.forwardProp(input)
        return self.layers[-1].outputs

    #Print output values of each node in each layer
    def printNet(self):
        for i in range(len(self.layers)):
            print(str(i) + " : " + str(self.layers[i].outputs))

    #Sigmoid activation function (1 / 1 + e^-x)
    def sigmoid(x):
        #Avoid overflow from following operation
        x = np.array(x, dtype=np.float128)
        return 1 / (1 + np.exp(-x))

    #Derivative of the sigmoid function w.r.t. weight (f(x)(1 - f(x)))
    def sigmoidDerWeight(x):
        return NeuralNet.sigmoid(x) * (1 - NeuralNet.sigmoid(x))

    #Weights * inputs + bias (W*I + b)
    def weightedSum(weights, inputs, bias):
        return weights.dot(inputs.reshape(weights.shape[-1])).reshape(bias.shape) + bias

    #Derivative of cost function w.r.t. activation output (O - E)
    def costDerActivation(output, expected):
        return output - expected.reshape(output.shape)

    #Derivative of cost function w.r.t. input sum from last layer ((O - E) * f(x)(1 - f(x))) - error signal for last layer
    def costDerWeightedSum(output, expected, inputs):
        temp = NeuralNet.costDerActivation(output, expected)
        return temp * NeuralNet.sigmoidDerWeight(inputs).reshape(temp.shape)



    #Derivative of cost w.r.t. weights
    def costDerWeights(errorSignals, prevLayerOut):
        return np.dot(errorSignals, np.atleast_2d(prevLayerOut.T))



    #Get error signals for hidden layers
    def errorSignalHidden(weightsNext, errorNext, inputs):
        return weightsNext.T.dot(errorNext) * NeuralNet.sigmoidDerWeight(inputs)

    #Error for training set
    def error(self, inputs, expected):
        cost = np.array([])
        for i in range(len(inputs)):
            outputs = np.array(self.layers[-1].outputs, dtype=np.float128)
            self.forwardProp(inputs[i])
            cost = np.append(cost, (outputs - expected[i]) ** 2)
        return np.sum(cost) / 2



if __name__ == '__main__':
    #2x2x1 neural network trained on all possible inputs of XOR function
    network = NeuralNet([2, 2, 1])
    xTrain, yTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])

    #After doing some research I found this to be a good starting point for
    #the number of epochs and the learning rate
    epochs = 2500
    learnRate = 0.8

    #Train for XOR function
    network.train(xTrain, yTrain, epochs, learnRate)
    x1 =  np.linspace(0,1,11) * np.ones((11, 1))
    x2 = x1.T
    x1, x2 = x1.flatten(), x2.flatten()
    out = np.array([])

    #Sub plots
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Neural Networks', fontsize=16)

    #XOR Scatter plot
    for x, y in zip(x1, x2):
            out = np.append(out, network.predict(np.array([x, y])))
    axs[0].set_title("XOR Function")
    axs[0].set_xlabel("Input 1")
    axs[0].set_ylabel("Input 2")
    axs[0].scatter(x1, x2, c=out, s=10, cmap="gray_r")



    #Random training data for X^2
    x = 20 * (2 * np.random.random_sample(1000) - 1)
    temp = 2 * np.random.random_sample(1000) - 1
    y = (x + temp) ** 2
    xTrain = np.vstack((x, y)).T
    yTrain = np.array([])

    #Labels for training data
    for x in xTrain:
        if (x[0] ** 2) > x[1]:
            yTrain = np.append(yTrain, [1, 0])

        else:
            yTrain = np.append(yTrain, [0, 1])
    yTrain = yTrain.reshape(1000, 2)

    #After some experimentation I found this to be a fairly good architecture
    network1 = NeuralNet([2,8,2])

    #I found these values to produce a pretty good fit to the x^2 function
    epochs = 2400
    learnRates = [0.001 * (2 ** x) for x in range(4)]
    i = 0
    for learnRate in learnRates:
        network1.train(xTrain, yTrain, epochs, learnRate)

        #Test data to plot
        x1 =  np.arange(-10,10,0.25) * np.ones((80, 1))
        x2 = x1.T
        x1, x2 = x1.flatten(), x2.flatten()
        out = np.array([])
        for x, y in zip(x1, x2):
                #Convert 2-d output to scalar and distribute values
                out = np.append(out, np.std(network1.predict(np.array([x, y]))))

        #Normalize output
        min, max = out.min(), out.max()
        out = (out - min) / (max - min)
        if i == 0:
            temp = out
            i += 1
        temp += out

    #Get average value for 10 different learning rates
    out = temp / len(learnRates)

    #Graph
    axs[1].set_title("X^2 Function")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].scatter(x1, x2, c=out, s=5, cmap="Blues_r")
    axs[1].set_xlim([-10, 10])
    axs[1].set_ylim([-10, 10])
    plt.show()
