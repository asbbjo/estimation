import numpy as np

setosa = []
versicolor = []
virginica = []

with open("Iris_project/Classification Iris/Iris_TTT4275/iris.data",'r') as reader:
    lines = reader.readlines()
    for el in lines:
        line = el.split(',')
        flower = line.pop(-1)
        data = np.loadtxt(line)
        data = data.tolist()
        if (flower == "Iris-setosa\n"):
            setosa.append(data)
        elif (flower == "Iris-versicolor\n"):
            versicolor.append(data)
        elif (flower == "Iris-virginica\n"):
            virginica.append(data)
        
#True class labels
y_true = [0,1,2]

#Weights
start_weight = 0.2
w0 = start_weight * np.ones(5)
w1 = start_weight * np.ones(5)
w2 = start_weight * np.ones(5)

#Predicted class labels
y_pred = np.zeros(30)
alpha0 = 0.04
alpha1 = 0.02
alpha2 = 0.01

#Iteration of setosa weights
for i in range(30):
    data_setosa = setosa[i]
    y_pred = 0

    #Calculate the predicted class label with weights
    for k in range(4):
        y_pred += w0[k+1] * data_setosa[k]
    y_pred += w0[0]

    #Update the weights
    w0[0] = w0[0] - alpha0 * (y_pred - y_true[0])
    for j in range(4):
        w0[j+1] = w0[j+1] - alpha0 * (y_pred - y_true[0]) * data_setosa[j]



#Iteration of versicolor weights
for i in range(30):
    data_versicolor = versicolor[i]
    y_pred = 0

    #Calculate the predicted class label with weights
    for k in range(4):
        y_pred += w0[k+1] * data_versicolor[k]
    y_pred += w1[0]

    #Update the weights
    w1[0] = w1[0] - alpha1 * (y_pred - y_true[1])
    for j in range(4):
        w1[j+1] = w1[j+1] - alpha1 * (y_pred - y_true[1]) * data_versicolor[j]



#Iteration of viginica weights
for i in range(30):
    data_virginica = virginica[i]
    y_pred = 0

    #Calculate the predicted class label with weights
    for k in range(4):
        y_pred += w2[k+1] * data_virginica[k]
    y_pred += w2[0]

    #Update the weights
    w2[0] = w2[0] - alpha2 * (y_pred - y_true[2])
    for j in range(4):
        w2[j+1] = w2[j+1] - alpha2 * (y_pred - y_true[2]) * data_virginica[j]


decision = np.zeros(3)

#test setosa (w0 weights for all)
for i in range(30,50,1):
    #setosa
    test_setosa = setosa[i]
    decision_sum = 0
    for k in range(4):
        decision_sum += w0[k+1] * test_setosa[k]
    decision_sum += w0[0]
    decision[0] = decision_sum

    #versicolor
    test_versicolor = versicolor[i]
    decision_sum = 0
    for k in range(4):
        decision_sum += w0[k+1] * test_versicolor[k]
    decision_sum += w0[0]
    decision[1] = decision_sum

    #virginica
    test_virginica = virginica[i]
    decision_sum = 0
    for k in range(4):
        decision_sum += w0[k+1] * test_virginica[k]
    decision_sum += w0[0]
    decision[2] = decision_sum

    print(np.argmax(decision))
    input()






