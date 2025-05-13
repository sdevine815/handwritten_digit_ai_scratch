import struct
import random
import math

class NeuralNetwork:

  #constructor for the neural network
  def __init__(self, input_size, hidden_size, output_size):
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    #initialize weights and biases with random small values
    self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
    self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
    
    self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
    self.b2 = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
  
  def feedforward(self, x):
    #activation layer: z1 = W1 * x + b1
    self.z1 = [sum(x[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j] for j in range(self.hidden_size)]

    #activation function
    self.a1 = [self.__sigmoid(z) for z in self.z1]

    #activation layer 2
    self.z2 = [sum(self.a1[i] * self.w2[i][j] for i in range(self.hidden_size)) + self.b2[j] for j in range(self.output_size)]
    self.a2 = [self.__sigmoid(z) for z in self.z2]
    
    return self.a2
  
  def backpropagate(self, x, y, lr=0.1):
    #1. Feedforward
    #2. Compute error: sigmoid'(z2) * (a2-y)
    #3. Propogate the erros back to hidden layer
    #4. Updates weights and biases uses lr (learning rate)
    
    #Forward pass
    self.feedforward(x)
  
    #Calculate output layer error
    output_errors = [(self.a2[i] - y[i]) * self.__sigmoid_derivative(self.z2[i]) for i in range(self.output_size)]
    
    #Calculate hidden layer error
    hidden_errors = [
      sum(output_errors[k] * self.w2[j][k] for k in range(self.output_size)) * self.__sigmoid_derivative(self.z1[j])
      for j in range(self.hidden_size)
    ]
  
    # Update weights and biases for w2, b2
    for i in range(self.hidden_size):

      for j in range(self.output_size):
        self.w2[i][j] -= lr * output_errors[j] * self.a1[i]
        
      for j in range(self.output_size):
        self.b2[j] -= lr * output_errors[j]
    
    # Update weights and biases for w1, b1
    for i in range(self.input_size):
      
      for j in range(self.hidden_size):
        self.w1[i][j] -= lr * hidden_errors[j] * x[i]
        
      for j in range(self.hidden_size):
        self.b1[j] -= lr * hidden_errors[j]
  
  def train(self, data, labels, epochs=1, lr=0.1):
    #training runs the prediction algorithm while also running the backpropagation to update the data
    for epoch in range(epochs):
      for x, y_label in zip(data, labels):
        #creates a zero vector of the 10 possible values
        y = [0.0] * self.output_size
        y[y_label] = 1.0
        self.backpropagate(x, y, lr)
        print("Epoch", epoch + 1, "completed.")
  
  def predict(self, x):
    output = self.feedforward(x)
    return output.index(max(output))

  #sigmoid function represented by 1 / ( 1 + e^-x )
  def __sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
  
  #derivatve of the sigmoid function
  def __sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)
