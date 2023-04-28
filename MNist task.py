import numpy as np
import matplotlib.pyplot as plt
import struct
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


"""

Read data from from files

"""

train_image_file = 'train_images.bin'
test_image_file = 'test_images.bin'
train_label_file = 'train_labels.bin'
test_label_file = 'test_labels.bin'


with open(train_image_file,'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    train_images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_images = train_images.reshape((size, nrows, ncols))
    
with open(test_image_file,'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    test_images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_images = test_images.reshape((size, nrows, ncols))

with open(train_label_file,'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_labels = train_labels.reshape((size,)) # (Optional)

with open(test_label_file,'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_labels = test_labels.reshape((size,)) # (Optional)


"""

Functions for clustering, error rate, mode and for the classification (NN, NN-clustering and kNN-clustering)

"""


def clustering(training_images, training_labels, data_length, M=64):
    """
    calculate the clustering matrix (640 x 784)
    """
    number_of_digits = 10           # #of digits
    digit_matrix_num = 784          #28x28 pixels
    digit_tot_clustering = 640      #64 clusters x 10 digits
    
    #sort the data classwise from 0 to 9
    numCount = [0]*number_of_digits
    images = training_images[:data_length]
    labels = training_labels[:data_length]
    for i in range(len(labels)):
        numCount[labels[i]] += 1
    
    #create the sorted vectors
    sortedLabels = np.argsort(labels)
    sortedImages = np.zeros_like(images)
    
    for i in range(len(labels)):
        sortedImages[i] = images[sortedLabels[i]]
    
    #cluster the images classwise. KMeans cluster the dataset in to M groups
    all_clusters = np.zeros((number_of_digits,M,digit_matrix_num))
    shapedImages = sortedImages.flatten().reshape(data_length,digit_matrix_num)
    start = 0
    end = 0
    print('Prepare clustering for each of the ten numbers:')
    for i in range(number_of_digits):
        print(i)
        end += numCount[i]
        cluster = KMeans(n_clusters=M, random_state=0).fit(shapedImages[start:end]).cluster_centers_
        all_clusters[i] = np.absolute(np.around(cluster))
        start = end
        
        
    #obtaining a 640x784 array. First 64th rows are 0's, next are 1's, and so on
    final_cluster = all_clusters.flatten().reshape(digit_tot_clustering,digit_matrix_num)
    return final_cluster
    



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


def mode(row):
    """
    calculate the mode for all the numbers in a vector
    """
    digits = [0]*10
    for num in row:
        digits[int(num)] += 1
    mode_digit = np.argmax(digits)
    return mode_digit


def classify_NN(train_images_data, train_labels_data, test_images_data, test_labels_data, train_image_number, test_image_number):
    """
    classify the test images with NN, and find the confusion matrix
    """
    digit_matrix_num = 784          #28x28 pixels

    #prepare the data sets for desired lengths and dimetions
    train_images_data = train_images_data[:train_image_number]
    test_images_data = test_images_data[:test_image_number]
    train_images_data = train_images_data.flatten().reshape(train_image_number,digit_matrix_num)
    test_images_data = test_images_data.flatten().reshape(test_image_number,digit_matrix_num)

    #find the distances and predict the digits in the image
    predicted = []
    for i in range(test_image_number):
        print(i)
        dist = []
        for j in range(train_image_number):
            dist.append(np.linalg.norm(test_images_data[i]-train_images_data[j]))
        
        predicted.append(train_labels_data[np.argmin(dist)])

    #calculate the confusion matrix and the error
    cm = confusion_matrix(test_labels_data[:test_image_number], predicted)
    error = error_rate(cm)

    return cm, error


def clustering_NN(train_images_data, train_labels_data, test_images_data, test_labels_data, train_image_number, test_image_number):
    """
    classify the test images with clustering NN, and find the confusion matrix
    """
    digit_matrix_num = 784          #28x28 pixels
    digit_tot_clustering = 640      #64 clusters x 10 digits
    
    #cluster the training set
    clustered = clustering(train_images_data, train_labels_data, train_image_number)
    
    #prepare the data sets for desired lengths and dimetions
    test_images_data = test_images_data[:test_image_number]
    test_images_data = test_images_data.flatten().reshape(test_image_number,digit_matrix_num)

    #find the distances of the nearest neighbor and predict the digits in the image
    predicted = []
    for i in range(test_image_number):
        print(i)
        dist = []
        for j in range(digit_tot_clustering):
            dist.append(np.linalg.norm(test_images_data[i]-clustered[j,:]))
        
        predicted.append(int(np.argmin(dist)//64))

    #calculate the confusion matrix and the error
    cm = confusion_matrix(test_labels_data[:test_image_number], predicted)
    error = error_rate(cm)

    return cm, error



def clustering_kNN(train_images_data, train_labels_data, test_images_data, test_labels_data, train_image_number, test_image_number, K=7):
    """
    classify the test images with clustering kNN, and find the confusion matrix
    """
    digit_matrix_num = 784          #28x28 pixels
    digit_tot_clustering = 640      #64 clusters x 10 digits
    
    #cluster the training set
    clustered = clustering(train_images_data, train_labels_data, train_image_number)
    
    #prepare the data sets for desired lengths and dimetions
    test_images_data = test_images_data[:test_image_number]
    test_images_data = test_images_data.flatten().reshape(test_image_number,digit_matrix_num)
    
    #find the distances of the k nearest neighbors, find the mode and predict the digits in the image
    predicted = []
    dist_matrix = np.zeros((test_image_number,K))
    for i in range(test_image_number):
        print(i)
        dist = []
        for j in range(digit_tot_clustering):
            dist.append(np.linalg.norm(test_images_data[i]-clustered[j,:]))
        for k in range(K):
            dist_matrix[i,k] = int(np.argmin(dist)//64)
            dist.pop(np.argmin(dist))
        predicted.append(mode(dist_matrix[i,:]))

    #calculate the confusion matrix and the error
    cm = confusion_matrix(test_labels_data[:test_image_number], predicted)
    error = error_rate(cm)

    return cm, error




"""

Start of the code. Uncomment and change i) train_number and ii) test_number

"""

#cm, error = classify_NN(train_images, train_labels, test_images, test_labels, 60000, 10000) 
#cm, error = clustering_NN(train_images, train_labels, test_images, test_labels, 60000, 10000) 
#cm, error = clustering_kNN(train_images, train_labels, test_images, test_labels, 60000, 10000)
print(cm, error)





















