import numpy as np
import scipy.special as sc
from sklearn.metrics import confusion_matrix

# calculating the gradient of the matrix W times the MSE
def grad_W_MSE_k(gk, tk, xk):
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
training_num = 30 
test_num = 20

# first round: training uses the first 30, testing the last 20
data_training = np.concatenate([setosa[:training_num], versicolor[:training_num], virginica[:training_num]])
data_test = np.concatenate([setosa[-test_num:], versicolor[-test_num:], virginica[-test_num:]])

# second round: training uses the last 30, testing the first 20
data_training = np.concatenate([setosa[-training_num:], versicolor[-training_num:], virginica[-training_num:]])
data_test = np.concatenate([setosa[:test_num], versicolor[:test_num:], virginica[:test_num:]])

# create a vector with the correct corresponding labels 
t_training = np.zeros((90, 3, 1))
t_training[:30] = np.array([[1],[0],[0]])
t_training[30:60] = np.array([[0],[1],[0]])
t_training[60:] = np.array([[0],[0],[1]])

# create matrix W (3,5) 
# [w1a w1b w1c w1d w10]
# [w2a w2b w2c w2d w20]
# [w3a w3b w3c w3d w30]

np.random.seed(100)
W = np.zeros((3, 5))
W = np.random.randn(3, 5)

######----- train the classifier -----######
# xk: (5,1), zk: (3,1), gk: (3,1)

alpha = 0.001
tolerance = 0.4
condition = True
num_iterations = 0

while condition:
    grad_W_MSE = 0  
    for i in range(3*training_num):

        xk_training = np.array([data_training[i]]).T
        zk_training = np.dot(W, xk_training)
        gk_training = sc.expit(zk_training)

        grad_W_MSE += grad_W_MSE_k(gk_training, t_training[i], xk_training)

    W = W - alpha*grad_W_MSE
    condition = np.linalg.norm(grad_W_MSE) >= tolerance
    num_iterations += 1
print('Number of iterations to converge for the training set:', num_iterations)
print('Using alpha=', alpha, ' and a tolerance=', tolerance, ' for the norm of grad_W_MSE')
    
######----- test the classifier -----######
# using the training set
g_predicted_training = []
for i in range(3*training_num):
    
    xk_training = np.array([data_training[i]]).T
    zk_training = np.dot(W, xk_training)
    gk_training = sc.expit(zk_training)

    g_predicted_training.append(np.argmax(gk_training) + 1)

# using the test set
g_predicted_test = []
for i in range(3*test_num):
    
    xk_test = np.array([data_test[i]]).T
    zk_test = np.dot(W, xk_test)
    gk_test = sc.expit(zk_test)

    g_predicted_test.append(np.argmax(gk_test) + 1)

# calculate the confusion matrix for the training and test set
g_true_training = [1]*training_num + [2]*training_num + [3]*training_num
g_true_test = [1]*test_num + [2]*test_num + [3]*test_num

cm_training = confusion_matrix(g_true_training, g_predicted_training)
cm_test = confusion_matrix(g_true_test, g_predicted_test)

# normalize confusion matrices to percentages
cm_norm_training = np.round(cm_training.astype('float') / cm_training.sum(axis=1)[:, np.newaxis], decimals=3)
cm_norm_test = np.round(cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis], decimals=3)

print('Confusion matrix for the training set:\n', cm_training, '\n\n', cm_norm_training)
print('Confusion matrix for the test set:\n', cm_test, '\n\n', cm_norm_test)