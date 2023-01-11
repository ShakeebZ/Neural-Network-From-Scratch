import numpy as np

def main():
    print("Testing Neurons:")
    weights = np.array([0,1])
    bias = 0
    x = np.array([2, 3])
    n = Neuron(weights, bias)
    h=n.feedforward(x)
    print(h)
    o = n.feedforward([h, h])
    print(o)
    print("Success!")
    print("Testing Neural Network:")
    network = NeuralNetwork()
    print(network.feedforward(x))
    print("Success!")

def sigmoid(x):
    """Returns the output of the sigmoid activation function

    Args:
        x: Input to be passed into Activation Function

    Returns:
        Activation Function: Function that bounds input in [0,1] 
        and returns the output as the feedforward
    """
    return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
    return np.dot(sigmoid(x), (1-sigmoid(x)))

def stochastic_gradient_descent(weight, bias, calculation, learning_rate):
    d_L_d_pred, d_pred_d_h, d_h_d_w = back_propogation()
    weight -= learning_rate*(np.dot(d_L_d_pred, d_pred_d_h, d_h_d_w))

def back_propogation():
    print("Hello World")

def mean_Squared_Error_Loss(calculated, predicted):
    return ((calculated - predicted)**2).mean()

#Calculate Change in Loss due to a change in weight.
#This is the partial derivative of Loss with respect to weight
#This is also equal to change in Loss due to change in prediction * change in prediction due to change in weight
#Change in Loss due to change in prediction is the derivative of mean_Squared_Error_Loss

#For Weight1, change in prediction due to change in weight1 is equal to change in prediction due to change in output of h1 * change in output of h1 due to change in Weight1


class Neuron:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    
    def feedforward(self, inputs):
        """Weigh the inputs, add the bias, use the activation function, and return the output

        Args:
            inputs : input coming into the neuron
        """
        weightedInput = np.dot(self.weights, inputs) + self.bias
        return sigmoid(weightedInput)

class NeuralNetwork:
    """A Neural Network with:
        - 2 inputs
        - 1 hidden layer with 2 Neurons
        - 1 Output layer with 1 Neuron
        
        Neurons all share same weight and bias:
        - Weight = [0,1]
        - Bias = 0
    """
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
    
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
    
    def feedforward(self, input):
        """Returns output to the Neural Network

        Args:
            input : Input to the Neural Network
        """
        h1_output = self.h1.feedforward(input)
        h2_output = self.h2.feedforward(input)
        
        o1_output = self.o1.feedforward(np.array([h1_output, h2_output]))
        
        return o1_output
    
        

if __name__ == "__main__":
    main()