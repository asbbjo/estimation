import numpy as np
import scipy.special as sc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

def make_x_data(setosa: np.ndarray, versicolor: np.ndarray, virginica: np.ndarray):
    """
    create the x vector from the compendium by adding 1 at the end
    """
    setosa_x = np.array([np.append(row, 1) for row in setosa])
    versicolor_x = np.array([np.append(row, 1) for row in versicolor])
    virginica_x = np.array([np.append(row, 1) for row in virginica])
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
    dataset: training or test
    num: training_num or test_num

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

    only needs a list of predicted values

    returns a confusion matrix and a normalized confusion matrix (in %) 
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
    total_predictions = np.sum(confusion_matrix)
    incorrect_predictions = total_predictions - np.trace(confusion_matrix)

    error_rate = incorrect_predictions / total_predictions
    error_rate = round(error_rate, 3)
    return error_rate

def print_results(cm: np.ndarray, cm_norm: np.ndarray, string: str):
    """
    print confusion matrix, normalized confusion matrix and error rate
    string: training or test
    """
    print('---', string.upper(),' SET:')
    print('Confusion matrix:\n', cm, '\n\n', cm_norm)
    print('\nError rate:', error_rate(cm))
    print()

##########---------- FOR TASK 2 ----------##########

def make_feature_data(data: np.ndarray):
    """
    returns a feature matrix:

    [sepal_length, sepal_width, petal_length, petal_width]

    where each of the features is a list
    """
    sepal_length = [element[0] for element in data]
    sepal_width  = [element[1] for element in data]
    petal_length = [element[2] for element in data]
    petal_width  = [element[3] for element in data]
    data_vector = [sepal_length, sepal_width, petal_length, petal_width]
    return data_vector

def plot_histogram(feature_data_matrix: np.ndarray, normalized=False, num_bins=10, opacity=0.5):
    """
    plot histogram

    default: not normalized
    """
    for i in range(len(feature_data_matrix)):
        plt.subplot(2, 2, i+1)
        plt.hist(feature_data_matrix[i][0:50], bins=num_bins, alpha=0.5, color='red', label='Setosa')
        plt.hist(feature_data_matrix[i][50:100], bins=num_bins, alpha=0.5, color='green', label='Versicolor')
        plt.hist(feature_data_matrix[i][100:150], bins=num_bins, alpha=0.5, color='blue', label='Virginica')
        if (normalized):
            plt.axis([0, 8, 0, 30])
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

def remove_features(data_set, alpha, tolerance):
    """
    removes 1 of the features in increasing order

    prints the results
    """
    xdata_setosa, xdata_versicolor, xdata_virginica = make_x_data(setosa, versicolor, virginica)
    for i in range(4):
        print('----- Using ', str(4-i), ' of the features -----')

        x3_training, x3_test = make_training_and_test_data(xdata_setosa, xdata_versicolor, xdata_virginica, data_set)
        W3 = train_classifier(alpha, tolerance, x3_training, num_cols_W=5-i, training_labels=t_labels)

        g3_pred_training = test_classifier(W3, x3_training, training_num)
        g3_pred_test = test_classifier(W3, x3_test, test_num)

        cm3_training, cm3_norm_training = calculate_confusion_matrix(g3_pred_training)
        cm3_test, cm3_norm_test = calculate_confusion_matrix(g3_pred_test)
        print_results(cm3_training, cm3_norm_training, "training")
        print_results(cm3_test, cm3_norm_test, "test")

        xdata_setosa, xdata_versicolor, xdata_virginica = delete_column(xdata_setosa, 0), delete_column(xdata_versicolor, 0), delete_column(xdata_virginica, 0)


def delete_column(matrix: np.ndarray, delete_index: int):
    """
    delete the element of delete_index for every list inside the larger list
    returns the matrix
    """
    return np.array([np.delete(row, delete_index) for row in matrix])

def print_text(text):
    print()
    print('----------' + text + '----------')
    print()

# variables to change:
###################################################################################
###################################################################################
alpha = 0.002
tolerance = 0.5    
###################################################################################
###################################################################################

string = 'You can change the values for alpha and the tolerance of how much the norm of grad_W_MSE (change in W divided by alpha) can change when updating the weight matrix W, in line 238 and 239.'
string += '\nStop the program to change the values, or press any key to continue the program with default values.'
print('##########---------- ----------##########')
print('Program started')
input(string)
print()

##########----------                    ----------##########
##########----------   RUNNING TASK 1   ----------##########
##########----------                    ----------##########


training_num = 30
test_num = 20

data_set_to_use = 1
for i in range(2):
    # read and preprepare data 
    setosa, versicolor, virginica, data = read_data("Classification Iris/Iris_TTT4275/iris.data")
    x_setosa, x_versicolor, x_virginica = make_x_data(setosa, versicolor, virginica)
    training_data, test_data = make_training_and_test_data(x_setosa, x_versicolor, x_virginica, data_set_to_use)
    t_labels = make_labels()

    # training and testing classifier
    W_matrix = train_classifier(alpha, tolerance, training_data, num_cols_W=5, training_labels=t_labels)
    g_pred_training = test_classifier(W_matrix, training_data, training_num)
    g_pred_test = test_classifier(W_matrix, test_data, test_num)

    # calculating confusion matrix and error rate
    cm_training, cm_norm_training = calculate_confusion_matrix(g_pred_training)
    cm_test, cm_norm_test = calculate_confusion_matrix(g_pred_test)
    print_results(cm_training, cm_norm_training, "training")
    print_results(cm_test, cm_norm_test, "test")
    
    data_set_to_use += 1
    print_text('Changing data set')

##########----------                    ----------##########
##########----------   RUNNING TASK 2   ----------##########
##########----------                    ----------##########
input('Press any key to continue to task 2.')
data_set_to_use = 1

feature_matrix = make_feature_data(data)
plot_histogram(feature_matrix)
print_text('Removing features')
remove_features(data_set_to_use, alpha, tolerance)
