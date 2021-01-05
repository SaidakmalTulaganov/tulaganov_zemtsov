import numpy
import scipy.io as sc
from displayData import displayData
import matplotlib.pyplot as plt
from predict import predict

test_set = sc.loadmat('test_set.mat')
weights = sc.loadmat('weights.mat')

testX = test_set['X']
testY = numpy.int64(test_set['y'])
weight1 = weights['Theta1']
weight2 = weights['Theta2']

m = testX.shape[0]

indexes = numpy.random.permutation(m)
x = numpy.zeros((100,testX.shape[1]))

for i in range(100):
    x[i] = testX[indexes[i]]

displayData(x)

#предсказание нейронной сети
pre = predict(weight1, weight2, testX)
testY.ravel()

p = (pre == testY.ravel())
print(p)
res = numpy.mean(numpy.double(p))
print(res)

#rp = numpy.random.permutation(m) 

plt.figure() 
for i in range(5): 
 
    X2 = testX[indexes[i],:]     
    X2 = numpy.matrix(testX[indexes[i]]) 
 
    pred = predict(weight1, weight2, X2.getA())     
    pred = numpy.squeeze(pred)    
    pred_str = 'Neural Network Prediction: %d (digit %d)' % (pred, testY[indexes[i]])    
    displayData(X2, pred_str) 
 
    plt.close() 
     
mistake = numpy.where(pre != testY.ravel())[0]
qwerty = numpy.zeros((100,testX.shape[1]))
for i in range(100):
    qwerty[i] = testX[mistake[i]]
displayData(qwerty)