import struct
import random
import math

# ----------- STEP 1: Load MNIST files manually -----------

def load_mnist_images(filename):

  #opens file with 'rb': read binary
  with open(filename, 'rb') as f:

    #uses struct to unpack binary data, takes in 4 signed integers, starts reading at 16th byte
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    images = []

    #reads in the binary, converts to integer, divides by 255 to make value between 0 and 1
    for _ in range(num):
      img = [ord(f.read(1)) / 255 for _ in range(rows * cols)]
      images.append(img)
      return images

def load_mnist_labels(filename):
  with open(filename, 'rb') as f:

    #reads in 2 signed integers starting at the 8th byte
    magic, num = struct.unpack(">II", f.read(8))
    
    #reads in a label which corresponds to the correct label for the images
    labels = [ord(f.read(1)) for _ in range(num)]
    return labels

# ----------- STEP 2: Define Activation Functions -----------

#sigmoid function represented by 1 / ( 1 + e^-x )
def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

#derivatve of the sigmoid function
def sigmoid_derivative(x):
  sx = sigmoid(x)
  return sx * (1 - sx)

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
