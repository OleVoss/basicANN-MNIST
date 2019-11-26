import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    '''
    np.savetxt("weightsIH.csv", wIH, delimiter=",")
    np.savetxt("weightsHH.csv", wHH, delimiter=",")
    np.savetxt("weightsHO.csv", wHO, delimiter=",")
    '''
    return wIH, wHH, wHO


# init weights
np.random.seed(123)
wIH, wHH, wHO = initNewWeights()

lr = 0.5
# load data from csv
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

# train loop
for epoch in range(1):
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


right = 0
for i in range(len(labelsArrTest)):
    if forward(inputArrTest[i], wIH, wHH, wHO)["out3"].argmax() == labelsArrTest[i].argmax():
        right += 1
print("Accuracy over 10000 test samples: ", str((right/len(labelsArrTest))*100) + "%")