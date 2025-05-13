from network import NeuralNetwork
from mnist_load import load_mnist_images, load_mnist_labels

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
