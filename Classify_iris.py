import numpy as np
import scipy.special as sc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
np.random.seed(100)

##########----------                    ----------##########
##########---------- DEFINING FUNCTIONS ----------##########
##########----------                    ----------##########

##########---------- FOR TASK 1 ----------##########

def read_data(filename: str):
    """
    read the data from the 3 flowers into setosa, versicolor, virginica and data (which contains all 3)
    """
    data = np.loadtxt(filename, delimiter=",", usecols=(0, 1, 2, 3))
    setosa = data[0:50]
    versicolor = data[50:100]
    virginica = data[100:150]
    return setosa, versicolor, virginica, data

def make_x_data(setosa: np.ndarray, versicolor: np.ndarray, virginica: np.ndarray, delete_index=slice(0,0)):
    """
    create the x vector from the compendium by adding 1 at the end
    default: deletes no elements

    when given a delete_index (int) it will delete that element (while also appending 1) (used for task 2)
    """
    setosa_x = np.array([np.append(np.delete(row, delete_index), 1) for row in setosa])
    versicolor_x = np.array([np.append(np.delete(row, delete_index), 1) for row in versicolor])
    virginica_x = np.array([np.append(np.delete(row, delete_index), 1) for row in virginica])
    return setosa_x, versicolor_x, virginica_x

def make_training_and_test_data(setosa_x: np.ndarray, versicolor_x: np.ndarray, virginica_x: np.ndarray, data_set: int, training_num = 30, test_num = 20):
    """
    make training and test data
    choose data set
    data_set == 1: training uses the first 30, testing the last 20
    data_set == 2: training uses the last 30, testing the first 20
    """
    if (data_set == 1):
        print('Using the data set of the FIRST round')
        data_training = np.concatenate([setosa_x[:training_num], versicolor_x[:training_num], virginica_x[:training_num]])
        data_test = np.concatenate([setosa_x[-test_num:], versicolor_x[-test_num:], virginica_x[-test_num:]])
    elif (data_set == 2):
        print('Using the data set of the SECOND round')
        data_training = np.concatenate([setosa_x[-training_num:], versicolor_x[-training_num:], virginica_x[-training_num:]])
        data_test = np.concatenate([setosa_x[:test_num], versicolor_x[:test_num], virginica_x[:test_num]])
    else:
        ValueError
    return data_training, data_test

def make_labels(training_num= 30):
    """
    make corresponding labels

    for 3 classes they are defined as [1 0 0]^T, [0 1 0]^T and [0 0 1]^T
    """
    t_training = np.zeros((3*training_num, 3, 1))
    t_training[:training_num] = np.array([[1],[0],[0]])
    t_training[training_num:2*training_num] = np.array([[0],[1],[0]])
    t_training[2*training_num:] = np.array([[0],[0],[1]])
    return t_training

def train_classifier(alpha: float, tolerance: float, dataset: np.ndarray, num_cols_W: int, training_labels: np.ndarray, training_num=30):
    """
    train the classifier

    needs a data set of matrix form (rows= training_num, cols= num_cols_W) 

    returns the matrix W
 
    """
    W = np.zeros((3, num_cols_W))
    W = np.random.randn(3, num_cols_W)
    condition = True
    num_iterations = 0
    print('Using alpha=', alpha, ' and a tolerance=', tolerance, 'for the norm of grad_W_MSE')
    while condition:
        grad_W_MSE = 0  
        for i in range(3*training_num):

            xk_training = np.array([dataset[i]]).T
            zk_training = np.dot(W, xk_training)
            gk_training = sc.expit(zk_training)

            grad_W_MSE += grad_W_MSE_k(gk_training, training_labels[i], xk_training)

        W = W - alpha*grad_W_MSE
        condition = np.linalg.norm(grad_W_MSE) >= tolerance
        num_iterations += 1
    print('Number of iterations to converge for the training set:', num_iterations)
    return W

def grad_W_MSE_k(gk, tk, xk):
    """
    calculate the gradient of the matrix W times the MSE
    """
    return np.dot(np.multiply(gk - tk, gk, 1 - gk), xk.T)

def test_classifier(W, dataset: np.ndarray, num: int):
    """
    test the classifier

    returns a list of the predicted values
    """
    g_predicted = []
    for i in range(3*num):
    
        xk = np.array([dataset[i]]).T
        zk = np.dot(W, xk)
        gk = sc.expit(zk)

        g_predicted.append(np.argmax(gk) + 1)
    return g_predicted

def calculate_confusion_matrix(g_predicted: list, training_num= 30, test_num= 20):
    """
    calculate the confusion matrix

    returns a confusion matrix and a normalized matrix (in %) 
    """
    if (len(g_predicted) == 3*training_num):
        g_true = [1]*training_num + [2]*training_num + [3]*training_num
    elif (len(g_predicted) == 3*test_num):
        g_true = [1]*test_num + [2]*test_num + [3]*test_num
    else:
        ValueError
    cm = confusion_matrix(g_true, g_predicted)

    # normalize confusion matrix to percentages
    cm_norm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    return cm, cm_norm

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

def print_results(cm: np.ndarray, cm_norm: np.ndarray, string: str):
    """
    print confusion matrix, normalized confusion matrix and error rate
    """
    print('---', string.upper(),' SET:')
    print('Confusion matrix:\n', cm, '\n\n', cm_norm)
    print('\nError rate:', error_rate(cm))
    print()

##########---------- FOR TASK 2 ----------##########



##########----------                    ----------##########
##########----------   RUNNING TASK 1   ----------##########
##########----------                    ----------##########

setosa, versicolor, virginica, data = read_data("Classification Iris/Iris_TTT4275/iris.data")
x_setosa, x_versicolor, x_virginica = make_x_data(setosa, versicolor, virginica)
training_data, test_data = make_training_and_test_data(x_setosa, x_versicolor, x_virginica, data_set=1)
t_labels = make_labels()
W_matrix = train_classifier(alpha=0.001, tolerance=0.4, dataset=training_data, num_cols_W=5, training_labels=t_labels)
g_pred_training = test_classifier(W=W_matrix, dataset=training_data, num=30)
g_pred_test = test_classifier(W=W_matrix, dataset=test_data, num=20)
cm_training, cm_norm_training = calculate_confusion_matrix(g_predicted=g_pred_training)
cm_test, cm_norm_test = calculate_confusion_matrix(g_predicted=g_pred_test)
print_results(cm=cm_training, cm_norm=cm_norm_training, string="training")
print_results(cm=cm_test, cm_norm=cm_norm_test, string="test")

##########----------                    ----------##########
##########----------   RUNNING TASK 2   ----------##########
##########----------                    ----------##########

# make histograms for each feture and class

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

