import numpy as np
import scipy.special as sc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
np.random.seed(50)

# calculating the gradient of the matrix W times the MSE
def grad_W_MSE_k(gk, tk, xk):
    return np.dot(np.multiply(gk - tk, gk, 1 - gk), xk.T)

def error_rate(confusion_matrix):
    # Calculate the total number of predictions
    total_predictions = np.sum(confusion_matrix)
    # Calculate the number of incorrect predictions
    incorrect_predictions = total_predictions - np.trace(confusion_matrix)
    # Calculate the error rate
    error_rate = incorrect_predictions / total_predictions
    error_rate = round(error_rate, 3)
    return error_rate

# read data from the 3 different flowers
data = np.loadtxt("Classification Iris/Iris_TTT4275/iris.data", delimiter=",", usecols=(0, 1, 2, 3))

setosa = data[0:50]
versicolor = data[50:100]
virginica = data[100:150]

# create vector x
setosa_x = np.array([np.append(row, 1) for row in setosa])
versicolor_x = np.array([np.append(row, 1) for row in versicolor])
virginica_x = np.array([np.append(row, 1) for row in virginica])

# read the data into a training and test set
training_num = 30 
test_num = 20

# CHOOSE DATA SET FOR TRAINING AND TEST
# first round: training uses the first 30, testing the last 20
print('Using the data set of the FIRST round')
data_training = np.concatenate([setosa_x[:training_num], versicolor_x[:training_num], virginica_x[:training_num]])
data_test = np.concatenate([setosa_x[-test_num:], versicolor_x[-test_num:], virginica_x[-test_num:]])

# second round: training uses the last 30, testing the first 20
# print('Using the data set of the SECOND round')
# data_training = np.concatenate([setosa_x[-training_num:], versicolor_x[-training_num:], virginica_x[-training_num:]])
# data_test = np.concatenate([setosa_x[:test_num], versicolor_x[:test_num], virginica_x[:test_num]])

# create a vector with the correct corresponding labels 
t_training = np.zeros((90, 3, 1))
t_training[:30] = np.array([[1],[0],[0]])
t_training[30:60] = np.array([[0],[1],[0]])
t_training[60:] = np.array([[0],[0],[1]])

# create matrix W (3,5) 
# [w1a w1b w1c w1d w10]
# [w2a w2b w2c w2d w20]
# [w3a w3b w3c w3d w30]

W = np.zeros((3, 5))
W = np.random.randn(3, 5)

######----- train the classifier -----######
# xk: (5,1), zk: (3,1), gk: (3,1)

alpha = 0.001
tolerance = 0.4
condition = True
num_iterations = 0
print('Using alpha=', alpha, ' and a tolerance=', tolerance, 'for the norm of grad_W_MSE')
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

print()
print('---TRAINING SET:')
print('Confusion matrix:\n', cm_training, '\n\n', cm_norm_training)
print('\nError rate:', error_rate(cm_training))
print('\n---TEST SET:')
print('Confusion matrix:\n', cm_test, '\n\n', cm_norm_test)
print('\nError rate:', error_rate(cm_test))

##########---------- TASK 2 ----------##########
# make histograms for each feture and class

# create data sets for each feature and class

sepal_length = [element[0] for element in data]
sepal_width  = [element[1] for element in data]
petal_length = [element[2] for element in data]
petal_width  = [element[3] for element in data]
data_vector = [sepal_length, sepal_width, petal_length, petal_width]

for i in range(len(data_vector)):
    plt.subplot(2, 2, i+1)
    plt.hist(data_vector[i][0:50], bins=10, alpha=0.5, color='red', label='Setosa')
    plt.hist(data_vector[i][50:100], bins=10, alpha=0.5, color='green', label='Versicolor')
    plt.hist(data_vector[i][100:150], bins=10, alpha=0.5, color='blue', label='Virginica')
    #plt.axis([0, 8, 0, 30])
    if (i == 0):
        plt.xlabel('Sepal length [cm]')
    elif (i == 1):
        plt.xlabel('Sepal width [cm]')
    elif (i == 2):
        plt.xlabel('Petal length [cm]')
    elif (i == 3):
        plt.xlabel('Petal width [cm]')
    plt.ylabel('Number of samples')
    plt.suptitle('Histogram for all features and classes')
    plt.legend()
plt.show()



