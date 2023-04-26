import numpy as np
import matplotlib.pyplot as plt
import struct

fname = 'train_images.bin'

with open(fname,'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

with open('train_labels.bin','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data_num = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_num = data_num.reshape((size,)) # (Optional)

for i in range(100):
    plt.imshow(data[i,:,:], cmap='gray')
    print(data_num[i])
    plt.show()