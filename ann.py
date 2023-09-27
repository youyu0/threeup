import numpy as np

class NeuralNetwork():
    def __init__(self, activation_function='sigmoid'):
        np.random.seed(1234)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        self.activation_function = activation_function

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def binary_step(self, x, threshold=0):
        return np.where(x >= threshold, 1, 0)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x)) 
        return exp_x / exp_x.sum()
    def softmax_derivative(self, x):
        return x

    def tanh(self, x):
        return np.tanh(x)
    def tanh_derivative(self, x):
        return 1 - x**2

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    def elu_derivative(self, x, alpha=1.0):
            return np.where(x > 0, 1, alpha * np.exp(x))

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            if self.activation_function == 'sigmoid':
                adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            elif self.activation_function == 'relu':
                adjustments = np.dot(training_inputs.T, error * self.relu_derivative(output)) 
            elif self.activation_function == 'leaky_relu':
                adjustments = np.dot(training_inputs.T, error * self.leaky_relu_derivative(output, alpha=0.01))
            elif self.activation_function == 'binary_step':
                adjustments = np.dot(training_inputs.T, error * self.binary_step(output))
            elif self.activation_function == 'softmax':
                adjustments = np.dot(training_inputs.T, error * self.softmax_derivative(output))
            elif self.activation_function == 'tanh':
                adjustments = np.dot(training_inputs.T, error * self.tanh_derivative(output))
            elif self.activation_function == 'elu':
                adjustments = np.dot(training_inputs.T, error * self.elu_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        if self.activation_function == 'sigmoid':
            return self.sigmoid(np.dot(inputs, self.synaptic_weights))
        elif self.activation_function == 'relu':
            return self.relu(np.dot(inputs, self.synaptic_weights))
        elif self.activation_function == 'leaky_relu':
            return self.leaky_relu(np.dot(inputs, self.synaptic_weights), alpha=0.01)
        elif self.activation_function == 'binary_step':
            return self.binary_step(np.dot(inputs, self.synaptic_weights))
        elif self.activation_function == 'softmax':
            return self.softmax(np.dot(inputs, self.synaptic_weights))
        elif self.activation_function == 'tanh':
            return self.tanh(np.dot(inputs, self.synaptic_weights))
        elif self.activation_function == 'elu':
            return self.elu(np.dot(inputs, self.synaptic_weights), alpha=1.0)

if __name__ == '__main__':
    activation_functions = ['sigmoid', 'relu','leaky_relu', 'binary_step',  'softmax', 'tanh', 'elu']

    for activation_function in activation_functions:
        print(f" activation function:{activation_function}")
        nn = NeuralNetwork(activation_function)
        print('Synaptic weights before training: \n', nn.synaptic_weights)
        train_input = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
        train_output = np.array([[0, 1, 1, 0]]).T
        nn.train(training_inputs=train_input,
                 training_outputs=train_output,
                 iterations=10000)

        print(f'Synaptic weights after training: \n{nn.synaptic_weights}')

        new_input1 = float(input('Input 1: '))
        new_input2 = float(input('Input 2: '))
        new_input3 = float(input('Input 3: '))

        print('New situation: input data = ', new_input1, new_input2, new_input3)
        print('Output data :',nn.think(np.array([new_input1, new_input2, new_input3])))
        print('-' * 20)

   