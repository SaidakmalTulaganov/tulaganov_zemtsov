import numpy as np
import matplotlib.pyplot as plt
import scipy.io

a = 10
print('a =', a)

arr = np.array([1, 2, 3])
print('Массив \n', arr)

arr1 = [[1], [1, 2], [1, 2, 3]]
print('Матрица 1 \n', arr1)

arr2 = np.zeros((3, 2))
print('Матрица 2 \n', arr2)

arr3 = np.ones((2, 3))
print('Матрица 3 \n', arr3)

arr4 = np.ones((2, 3)) * 7
print('Матрица 4 \n', arr4)

arr5 = np.random.random((2, 3))
print('Матрица 5 \n', arr5)

arr6 = np.random.randint(2, 6, (5, 5))
print('Матрица 6 \n', arr6)

data = np.loadtxt('./test.txt', dtype=np.int32)
print(data)

data1 = scipy.io.loadmat('./var5.mat')

data1 = data1['n']
mx = np.max(data1)
print(mx)

mn = np.min(data1)
print(mn)

md = np.median(data1)
print(md)

mo = np.mean(data1)
print(mo)

ds = np.var(data1)
print(ds)

sk = np.std(data1)
print(sk)

plt.plot(data1)
plt.show()
mean = np.mean(data1) * np.ones(len(data1))
var = np.var(data1) * np.ones(len(data1))
plt.plot(data1, 'b-', mean, 'r-', mean - var, 'g--', mean + var, 'g--')
plt.grid()
plt.show()
plt.hist(data1, bins=20)
plt.grid()
plt.show()


def autocorrelate(a):
    n = len(a)
    cor = []

    for i in range(n // 2, n // 2 + n):
        a1 = a[:i + 1] if i < n else a[i - n + 1:]
        a2 = a[n - i - 1:] if i < n else a[:2 * n - i - 1]
        cor.append(np.corrcoef(a1, a2)[0, 1])
    return np.array(cor)


data1 = np.ravel(data1)
cor = autocorrelate(data1)
plt.plot(cor)
plt.show()

data2 = scipy.io.loadmat(r'C:\Users\saida\Downloads\pract\03\data\ND\var2.mat')
data2.keys()
data2 = data2['mn']

n = data2.shape[1]
corr_matrix = np.zeros((n, n))
for i in range(0, n):
    for j in range(0, n):
        col = data2[:, i]
        col2 = data2[:, j]
        corr_matrix[i, j] = np.corrcoef(col, col2)[0, 1]
np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(data2[:, 2], data2[:, 5], 'b.')
plt.grid()
plt.show()
