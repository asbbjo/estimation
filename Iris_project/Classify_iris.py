import numpy as np
import scipy.special as sc

# create data for the flowers
data = np.loadtxt("Iris_project/Classification Iris/Iris_TTT4275/iris.data", delimiter=",", usecols=(0, 1, 2, 3))

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
start_weigth_0 = 0.4

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

    #calculate Delta MSE
    N = 100
    setosa_target = [1,0,0]
    versicolor_target = [0,1,0]
    virginica_target = [0,0,1]
    alpha = 0.00001
    for k in range(N):
        #loss function to minimize
        loss1 = 0.5 * np.sum((g1 - setosa_target)**2)
        loss2 = 0.5 * np.sum((g2 - versicolor_target)**2)
        loss3 = 0.5 * np.sum((g3 - virginica_target)**2)

        #calculate and update the weights
        MSE1 = np.outer(np.sum(g1 - setosa_target) * g1 * np.sum(1 - g1), setosa_train.T)
        W = W - alpha*MSE1
        MSE2 = np.outer(np.sum(g2 - versicolor_target) * g2 * np.sum(1 - g2), versicolor_train.T)
        W = W - alpha*MSE2
        MSE3 = np.outer(np.sum(g1 - virginica_target) * g3 * np.sum(1 - g3), virginica_train.T)
        W = W - alpha*MSE3

print(loss1, loss2, loss3)
print(W)

# test the classifier
test_num = 20
for i in range(test_num):
    print('New iteration:')

    setosa_test = np.array([setosa[train_num + i]])
    print(np.argmax(np.dot(W,setosa_test.T)))

    versicolor_test = np.array([versicolor[train_num + i]])
    print(np.argmax(np.dot(W,versicolor_test.T)))

    virginica_test = np.array([virginica[train_num + i]])
    print(np.argmax(np.dot(W,virginica_test.T)))

    input()
    