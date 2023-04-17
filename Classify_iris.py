import numpy as np
import scipy.special as sc

# create data for the flowers
data = np.loadtxt("Classification Iris/Iris_TTT4275/iris.data", delimiter=",", usecols=(0, 1, 2, 3))

setosa = data[0:50]
versicolor = data[50:100]
virginica = data[100:150]

setosa = np.array([np.append(row, 1) for row in setosa])
versicolor = np.array([np.append(row, 1) for row in versicolor])
virginica = np.array([np.append(row, 1) for row in virginica])

# create matrix W
start_weigth_1 = 0.1
start_weigth_2 = 0.2
start_weigth_3 = 0.3
start_weigth_0 = 0.9

w1 = np.full(4, start_weigth_1)
w2 = np.full(4, start_weigth_2)
w3 = np.full(4, start_weigth_3)
w0 = np.full(3, start_weigth_0)

w = np.array([w1, w2, w3])
W = np.insert(w, 4, w0, axis=1)

# train the classifier
train_num = 30
for i in range(train_num):
    setosa_train = np.array([setosa[i]])
    versicolor_train = np.array([versicolor[i]])
    virginica_train = np.array([virginica[i]])

    z_setosa = np.dot(W, setosa_train.T)
    z_versicolor = np.dot(W, versicolor_train.T)
    z_virginica = np.dot(W, virginica_train.T)

    g1 = sc.expit(z_setosa)
    g2 = sc.expit(z_versicolor)
    g3 = sc.expit(z_virginica)