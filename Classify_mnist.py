import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import cProfile
from sklearn.metrics import confusion_matrix

##########----------                    ----------##########
##########---------- DEFINING FUNCTIONS ----------##########
##########----------                    ----------##########

##########---------- FOR TASK 1 ----------##########

def read_file(filename: str):
    """
    read .bin file
    """
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        if "images" in filename:
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        elif "labels" in filename:
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size,))
        return data

def error_rate(confusion_matrix):
    """
    calculate error rate
    """
    # total number of predictions
    total_predictions = np.sum(confusion_matrix)
    # number of incorrect predictions
    incorrect_predictions = total_predictions - np.trace(confusion_matrix)

    error_rate = incorrect_predictions / total_predictions
    error_rate = round(error_rate, 3)
    return error_rate

##########----------                    ----------##########
##########----------   RUNNING TASK 1   ----------##########
##########----------                    ----------##########

filenames = ["train_images.bin", "train_labels.bin", "test_images.bin", "test_labels.bin"]
filepaths = []
for name in filenames:
    filepaths.append(os.path.join(os.getcwd(), "Classification MNIST\MNist_ttt4275", name))

train_images = read_file(filepaths[0])
train_labels = read_file(filepaths[1])
test_images  = read_file(filepaths[2])
test_labels  = read_file(filepaths[3])

# for i in range(100):
#     plt.imshow(train_images[i,:,:])
#     print(train_labels[i])
#     plt.show()

nrows = 1000
ncols = 6000

distances = np.zeros((nrows, ncols))

for uno in range(nrows):
    for dos in range(ncols):
        distances[uno, dos] = np.linalg.norm(test_images[uno] - train_images[dos])


predictions = np.array([train_labels[np.argmin(distances[i])] for i in range(nrows)])

print(predictions)
print(test_labels[:nrows])

cm = confusion_matrix(test_labels[:nrows], predictions)
error = error_rate(cm)
print(cm, error)
