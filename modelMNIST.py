import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(X):
    return 1/(1+np.exp(-X))

def forward(X, wIH, wHH, wHO):
    outs = {}
    out1 = sigmoid(np.dot(X, wIH))
    out2 = sigmoid(np.dot(out1, wHH))
    out3 = sigmoid(np.dot(out2, wHO))

    outs = {"out1":out1, "out2":out2, "out3":out3}

    return outs

def grads(Y, outs, wIH, wHH, wHO, X):
    grads = {}
    eHO = outs["out3"] - Y
    eHH = np.dot(eHO * outs["out3"] * (1 - outs["out3"]), wHO.T)
    eIH = np.dot(eHH * outs["out2"] * (1 - outs["out2"]), wHH.T)

    gHO = eHO * outs["out3"] * (1 - outs["out3"]) * outs["out2"].T
    gHH = eHH * outs["out2"] * (1 - outs["out2"]) * outs["out1"].T
    gIH = eIH * outs["out1"] * (1 - outs["out1"]) * X.T
    
    grads ={"gHO":gHO, "gHH":gHH, "gIH":gIH}
    
    return grads

def initNewWeights():
    wIH = np.random.rand(784, 20)
    wHH = np.random.rand(20, 20)
    wHO = np.random.rand(20, 10)

    return wIH, wHH, wHO

def saveWeights(wIH, wHH, wHO):
    np.savetxt("weightsIH.csv", wIH, delimiter=",")
    np.savetxt("weightsHH.csv", wHH, delimiter=",")
    np.savetxt("weightsHO.csv", wHO, delimiter=",")

def plotPred(img, pred, count, index):
    x = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    plt.subplot(count, 2, index)
    plt.imshow(img, cmap="Greys_r")

    plt.subplot(count, 2, index+1)
    plt.bar(x, pred)
   

# init weights
np.random.seed(123)
wIH, wHH, wHO = initNewWeights()

lr = 0.5
# load data from csv
print("[load data]")
trainDf = pd.read_csv("mnist_train.csv", header=None)
testDf = pd.read_csv("mnist_test.csv", header=None)

# separate labels from features
inputRaw = np.array(trainDf.iloc[:,1:])
inputRawTest = np.array(testDf.iloc[:,1:])
labelsRaw = np.array(trainDf.iloc[:,0])
labelsRawTest = np.array(testDf.iloc[:,0])

# normalize features
inputRaw = (inputRaw - inputRaw.mean()) / (inputRaw.max() - inputRaw.min())
inputRawTest = (inputRawTest - inputRawTest.mean()) / (inputRawTest.max() - inputRawTest.min())



# numpy array for training data
inputArr = np.zeros([60000, 784])
labelsArr = np.zeros([60000, 10])

# numpy arrays for test data
inputArrTest = np.zeros([10000, 784])
labelsArrTest = np.zeros([10000, 10])

# one hot encoding e.g: 7 -> [0,0,0,0,0,0,0,1,0,0]
for i in range(60000):
    labelsArr[i][labelsRaw[i]] = 1
    inputArr[i] = inputRaw[i, :]
for i in range(10000):
    labelsArrTest[i][labelsRawTest[i]] = 1
    inputArrTest[i] = inputRawTest[i, :]

# plot image for testing
'''
img = inputArr[0].reshape([28,28])
plt.imshow(img, cmap="gray")
plt.show()
'''
epochs = 4
# train loop
for epoch in range(epochs):
    print("###Epoch {}###".format(epoch))
    runningLoss = 0
    for i in range(len(inputArr)):
        X = np.array(inputArr[i]).reshape([1,-1])
        Y = labelsArr[i].reshape([1,-1])

        outAll = forward(X, wIH, wHH, wHO)
        grad = grads(Y, outAll, wIH, wHH, wHO, X)
        wIH = wIH - lr * grad["gIH"]
        wHH = wHH - lr * grad["gHH"]
        wHO = wHO - lr * grad["gHO"]
        runningLoss += Y - outAll["out3"]
    print("Average loss: {}\n".format(np.round_(np.mean(runningLoss),2)))

right = 0
for i in range(len(labelsArrTest)):
    if forward(inputArrTest[i], wIH, wHH, wHO)["out3"].argmax() == labelsArrTest[i].argmax():
        right += 1
print("\nAccuracy over 10000 test samples: ", str(np.round_(right/len(labelsArrTest)*100,2)) + "%")

count = 5



for i in range(1,count*2+1,2):
    img = inputArrTest[random.randint(0,10000)]
    pred = forward(img, wIH, wHH, wHO)["out3"]

    plotPred(img.reshape(28,28), pred, count, i)

plt.show()