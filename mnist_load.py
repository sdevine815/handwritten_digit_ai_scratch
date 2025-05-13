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
