import numpy as np
import scipy.special as sc

setosa = []
versicolor = []
virginica = []

with open("Classification Iris/Iris_TTT4275/iris.data",'r') as reader:
    lines = reader.readlines()
    for el in lines:
        line = el.split(',')
        flower = line.pop(-1)
        data = np.loadtxt(line)
        data = data.tolist()
        data.append(1)
        if (flower == "Iris-setosa\n"):
            setosa.append(data)
        elif (flower == "Iris-versicolor\n"):
            versicolor.append(data)
        elif (flower == "Iris-virginica\n"):
            virginica.append(data)

start_weigth_1 = 0.1
start_weigth_2 = 0.2
start_weigth_3 = 0.3
start_weigth_0 = 0.9

w1 = np.full(4, start_weigth_1)
w2 = np.full(4, start_weigth_2)
w3 = np.full(4, start_weigth_3)
w0 = np.full(3, start_weigth_0)

w = np.array([np.transpose(w1), np.transpose(w2), np.transpose(w3)])
W = np.insert(w, 4, w0, axis=1)

x = setosa[0]
z = np.dot(W, x)
g = sc.expit(z[0])
print(x)


def grad_w_MSE_k(gk, tk, xk):
    return np.multiply()

