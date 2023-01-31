import numpy as np

def main():
    print("Testing Neurons:")
    weights = np.array([0,1])
    bias = 0
    x = np.array([2, 3])
    n = Neuron(weights, bias)
    h=n.feed_forward(x)
    print(h)
    o = n.feed_forward([h, h])
    print(o)
    print("Success!")
    print("Testing Neural Network:")
    network = SimpleNeuralNetwork()
    print(network.feed_forward(x))
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

def sigmoid_derivative(x):
    return np.dot(sigmoid(x), (1-sigmoid(x)))

#Calculate Change in Loss due to a change in weight.
#This is the partial derivative of Loss with respect to weight
#This is also equal to change in Loss due to change in prediction * change in prediction due to change in weight
#Change in Loss due to change in prediction is the derivative of mean_Squared_Error_Loss

#For Weight1, change in prediction due to change in weight1 is equal to change in prediction due to change in output of h1 * change in output of h1 due to change in Weight1

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    
    def feed_forward(self, inputs):
        """Weigh the inputs, add the bias, use the activation function, and return the output

        Args:
            inputs : input coming into the neuron
        """
        weightedInput = np.dot(self.weights, inputs) + self.bias
        return sigmoid(weightedInput)

class SimpleNeuralNetwork:
    """A Neural Network with:
        - 2 inputs
        - 1 hidden layer with 2 Neurons
        - 1 Output layer with 1 Neuron
        
        Neurons all share same weight and bias:
        - Weight = [0,1]
        - Bias = 0
    """
    def __init__(self):
        self.weights = np.empty(6)
        self.bias = np.empty(3)
        for i in range(6):
            self.weights[i] = np.random.normal()
        for i in range(3):
            self.bias[i] = np.random.normal()
    
        self.h1 = Neuron(self.weights[0:2], self.bias[0])
        self.h2 = Neuron(self.weights[2:4], self.bias[1])
        self.o1 = Neuron(self.weights[4:], self.bias[2])
    
    def feed_forward(self, input):
        """Returns output to the Neural Network

        Args:
            input : Input to the Neural Network
        """
        h1_output = self.h1.feed_forward(input)
        h2_output = self.h2.feed_forward(input)
        
        o1_output = self.o1.feed_forward(np.array([h1_output, h2_output]))
        
        return o1_output
    
    def back_propogation(self, input, true_value):
        """Back Propogates to find the partial derivatives of various aspects in the neural network to be used in computation of the required change in weights and bias

        Args:
            input : 1 x 2 array
            true_value : the true value of the output

        Returns:
            d_L_d_pred: The value that correlates to the partial derivative of the loss with respect to the prediction
            d_pred_d_h: An array of 2 values containing the partial derivatives of the prediction with respect to the output of the hidden layer
            d_h_d_w: An array of 6 values containing the partial derivatives of the output of the hidden layer with respect to the weights
        """
        h_output = np.empty(2)
        h_sum = np.empty(2)
        d_h_d_b = np.empty(3)
        h_sum[0] = np.dot(self.weights[0:2], input) + self.bias[0]
        h_sum[1] = np.dot(self.weights[2:4], input) + self.bias[1]
        h_output = sigmoid(h_sum)
        o_sum = np.dot(self.weights[4:], h_output) + self.bias[2]
        d_h_d_b[0] = sigmoid_derivative(h_sum[0])
        d_h_d_b[1] = sigmoid_derivative(h_sum[1])
        d_h_d_b[2] = sigmoid_derivative(o_sum)
        o_output = predicted = sigmoid(o_sum)
        d_L_d_pred = -2*(true_value - predicted).mean()
        d_pred_d_h = self.weights[4:] * sigmoid_derivative(o_output)
        d_h_d_w = np.empty(6)
        for i in range(1,5):
            if ((i % 2) == False):
                d_h_d_w[i-1] = input[0]*sigmoid_derivative(h_sum[0])
            else:
                d_h_d_w[i-1] = input[1]*sigmoid_derivative(h_sum[1])
        d_h_d_w[4] = h_sum[0] * sigmoid_derivative(o_sum)
        d_h_d_w[5] = h_sum[1] * sigmoid_derivative(o_sum)
        return d_L_d_pred, d_pred_d_h, d_h_d_w, d_h_d_b
        
        
    def stochastic_gradient_descent(self, calculation, true_value, learning_rate):
        d_L_d_pred, d_pred_d_h, d_h_d_w, d_h_d_b = SimpleNeuralNetwork.back_propogation(calculation, true_value)
        # for i in range(1,5):
        #     if ((i%2) == False):
        #         self.weights[i-1] -= learning_rate*d_L_d_pred*d_pred_d_h[0]*d_h_d_w[i-1]
        #     else:
                
        #         print("Hello World")
        self.weights[0] -= learning_rate*d_L_d_pred*d_pred_d_h[0]*d_h_d_w[0]
        self.weights[1] -= learning_rate*d_L_d_pred*d_pred_d_h[0]*d_h_d_w[1]
        self.weights[2] -= learning_rate*d_L_d_pred*d_pred_d_h[1]*d_h_d_w[2]
        self.weights[3] -= learning_rate*d_L_d_pred*d_pred_d_h[1]*d_h_d_w[3]
        self.weights[4] -= learning_rate*d_L_d_pred*d_h_d_w[4]
        self.weights[5] -= learning_rate*d_L_d_pred*d_h_d_w[5]
        self.bias[0] -= learning_rate*d_L_d_pred*d_pred_d_h[0]*d_h_d_b[0]
        self.bias[1] -= learning_rate*d_L_d_pred*d_pred_d_h[1]*d_h_d_b[1]
        self.bias[2] -= learning_rate*d_L_d_pred*d_h_d_b[2]
        
    def train(self, data, true_value, number_of_episodes):
        number_of_episodes = 1000
        for number in range(number_of_episodes):
            for x, y_true in zip(data, true_value):
                self.stochastic_gradient_descent(x, y_true, 0.1)
                #PUT IN CORRECT PAREMETERS INTO FUNCTION
            if (number + 1) % 10 == 0:
                prediction = np.apply_along_axis(self.feed_forward, 1, data)
                loss = mean_squared_error_loss(true_value, prediction)
                print("Episode %d, Mean Squared Error Loss: %.3f" % (number + 1, loss))
            
        
def mean_squared_error_loss(calculated, predicted):
    return ((calculated - predicted)**2).mean()
    
        

if __name__ == "__main__":
    main()