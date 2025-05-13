import struct
import random
import math

# ----------- STEP 1: Load MNIST files manually -----------

def load_mnist_images(filename):
  with open(filename, 'rb') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    images = []
    for _ in range(num):
      img = [ord(f.read(1)) / 255 for _ in range(rows * cols)]
      images.append(img)
      return images

def load_mnist_labels(filename):
  with open(filename, 'rb') as f:
    magic, num = struct.unpack(">II", f.read(8))
    labels = [ord(f.read(1)) for _ in range(num)]
    return labels

# ----------- STEP 2: Define Activation Functions -----------

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
  sx = sigmoid(x)
  return sx * (1 - sx)

# ----------- STEP 3: Define Neural Network -----------

class NeuralNetwork:
def __init__(self, input_size, hidden_size, output_size):
  #Initialize weights and biases with random small values
  self.input_size = input_size
  self.hidden_size = hidden_size
  self.output_size = output_size

  self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
  self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
  
  self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
  self.b2 = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

def feedforward(self, x):
  self.z1 = [sum(x[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j] for j in range(self.hidden_size)]
  self.a1 = [sigmoid(z) for z in self.z1]
  
  self.z2 = [sum(self.a1[i] * self.w2[i][j] for i in range(self.hidden_size)) + self.b2[j] for j in range(self.output_size)]
  self.a2 = [sigmoid(z) for z in self.z2]
  
  return self.a2

def backpropagate(self, x, y, lr=0.1):
  # Forward pass
  self.feedforward(x)

# Calculate output layer error
output_errors = [(self.a2[i] - y[i]) * sigmoid_derivative(self.z2[i]) for i in range(self.output_size)]

# Calculate hidden layer error
hidden_errors = [
sum(output_errors[k] * self.w2[j][k] for k in range(self.output_size)) * sigmoid_derivative(self.z1[j])
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
  for epoch in range(epochs):
    for x, y_label in zip(data, labels):
      y = [0.0] * self.output_size
      y[y_label] = 1.0
      self.backpropagate(x, y, lr)
      print("Epoch", epoch + 1, "completed.")

def predict(self, x):
  output = self.feedforward(x)
  return output.index(max(output))

# ----------- STEP 4: Train and Evaluate -----------

# You must download these files from the MNIST website:
# t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte, train-images.idx3-ubyte, train-labels.idx1-ubyte
train_images = load_mnist_images("train-images.idx3-ubyte")
train_labels = load_mnist_labels("train-labels.idx1-ubyte")
test_images = load_mnist_images("t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")

nn = NeuralNetwork(784, 32, 10)
nn.train(train_images[:1000], train_labels[:1000], epochs=5, lr=0.1) # using 1k samples for speed

# Test accuracy
correct = 0
for i in range(100):
  prediction = nn.predict(test_images[i])
  if prediction == test_labels[i]:
    correct += 1

print("Accuracy on 100 test samples:", correct, "%")
