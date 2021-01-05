import numpy as np

# вычисление отклика нейронной сети на входной пример 
from sigmoid import sigmoid

def predict(theta1, theta2, x):
    a1 = np.c_[np.ones(x.shape[0]), x]
    a2 = sigmoid(np.dot(a1, theta1.T))
    a3 = np.c_[np.ones(a2.shape[0]), a2]
    tmp = sigmoid(np.dot(a3, theta2.T))
    res = np.argmax(tmp, axis=1)+1     
    return res
    