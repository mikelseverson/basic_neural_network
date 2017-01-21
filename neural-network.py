from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the ranom number generator
        # So it generates the same number every time the program runs
        random.seed(1)

        #model a single neuron, with 3 input connections -> 1 output connection.
        #Assign random weights to a 3 x 1 matrix, with values between -1 to 1
        #With a mean of 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    #Sigmoid function, which describes an S curve
    # we pass weighted sum of the inputs through this function
    # to normalize between 0 and 1
    def __sigmoid(self, x):
        return 1 /(1 + exp(-x))
        
    #Helps us understand how confident we are
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            
            #Pass training set through neural network
            output = self.predict(training_set_inputs)

            error = training_set_outputs - output

            #multiply the error by the input and again by the gradient of sigmoid curve
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def predict(self, inputs):
            return self.__sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == "__main__":

    #initializing a single neruon nerual network
    neural_network = NeuralNetwork()

    print 'Random starting synaptic weights:'
    print neural_network.synaptic_weights

    #The training set. We have 4 examples, 
    #   each consist of:
    #     3 input  values
    #     1 output values
    training_set_inputs  = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[   0,         1,         1,         0    ]]).T

    # Train the neural network using the training set
    # 10,000 iterations making small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000000)

    print 'New synaptic weights after training: '
    print neural_network.synaptic_weights

    #Test the neural network
    print 'Predicting'
    print neural_network.predict(array([1, 0, 0]))
