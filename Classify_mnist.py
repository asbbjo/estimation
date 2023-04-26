import numpy as np
import matplotlib.pyplot as plt
import struct
import os

filenames = ["train_images.bin", "train_labels.bin", "test_images.bin", "test_labels"]
filepaths = []
for i in range(4):
    filepaths.append(os.path.join(os.getcwd(), "Classification MNIST\MNist_ttt4275", filenames[i]))

with open(filepaths[0],'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

with open(filepaths[1],'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data_num = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_num = data_num.reshape((size,)) # (Optional)

for i in range(100):
    plt.imshow(data[i,:,:], cmap='gray')
    print(data_num[i])
    plt.show()