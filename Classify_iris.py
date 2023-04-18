import numpy as np
import scipy.special as sc

# calculating the gradient of the matrix W times the MSE
def grad_w_MSE_k(gk, tk, xk):
    return np.dot(np.multiply(gk - tk, gk, 1 - gk), xk.T)

# read data from the 3 different flowers
data = np.loadtxt("Classification Iris/Iris_TTT4275/iris.data", delimiter=",", usecols=(0, 1, 2, 3))

setosa = data[0:50]
versicolor = data[50:100]
virginica = data[100:150]

# create vector x
setosa = np.array([np.append(row, 1) for row in setosa])
versicolor = np.array([np.append(row, 1) for row in versicolor])
virginica = np.array([np.append(row, 1) for row in virginica])

# read the data into a training and test set
# training uses the first 30, testing the last 20
training_num = 30 
test_num = 20

data_training = np.concatenate([setosa[:training_num], versicolor[:training_num], virginica[:training_num]])
data_test = np.concatenate([setosa[-test_num:], versicolor[-test_num:], virginica[-test_num:]])

# create a vector with the correct corresponding labels (1, 2, 3)
t_training = [1]*training_num + [2]*training_num + [3]*training_num
t_test = [1]*test_num + [2]*test_num + [3]*test_num

# create matrix W (3,5) 
# [w1a w1b w1c w1d w10]
# [w2a w2b w2c w2d w20]
# [w3a w3b w3c w3d w30]

np.random.seed(1)
W = np.random.randn(3, 5)

# train the classifier
# xk: (5,1), zk: (3,1), gk: (3,1)

grad_w_MSE = 0
alpha = 0.01
for j in range(3):
    for i in range(3*training_num):

        xk_training = np.array([data_training[i]]).T
        zk_training = np.dot(W, xk_training)
        gk_training = sc.expit(zk_training)

        grad_w_MSE += grad_w_MSE_k(gk_training, t_training[i], xk_training)
        g_predicted_training = np.argmax(gk_training) + 1

    W = W - alpha*grad_w_MSE
    print(W)
    print()

# test the classifier
# xk: (5,1), zk: (3,1), gk: (3,1)
for i in range(3*test_num):

    xk_test = np.array([data_test[i]]).T
    zk_test = np.dot(W, xk_test)
    gk_test = sc.expit(zk_test)

    g_predicted_test = np.argmax(gk_test) + 1
    print(gk_test,g_predicted_test, t_test[i])
    
