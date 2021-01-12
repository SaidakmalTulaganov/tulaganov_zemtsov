import numpy as np
import matplotlib.pyplot as plt
import scipy.io

a = 10
# переменная
print('a =', a)

arr = np.array([1, 2, 3])
# массив
print('Массив \n', arr)

arr1 = [[1], [1, 2], [1, 2, 3]]
# матрица с заданными значениями
print('Матрица 1 \n', arr1)

arr2 = np.zeros((3, 2))
# матрица с нулевыми значениями
print('Матрица 2 \n', arr2)

arr3 = np.ones((2, 3))
# матрица с нулевыми значениями
print('Матрица 3 \n', arr3)

arr4 = np.ones((2, 3)) * 7
# матрица с нулевыми значениями
print('Матрица 4 \n', arr4)

arr5 = np.random.random((2, 3))
print('Матрица 5 \n', arr5)

arr6 = np.random.randint(2, 6, (5, 5))
# матрица со случайными целочисленными значениями
print('Матрица 6 \n', arr6)

data = np.loadtxt('./test.txt', dtype=np.int32)
#импорт данных из файла и указываем что там целые числа
print(data)

#загрузка одномерных данных
array_var5 = scipy.io.loadmat('./var5.mat')
array_var5 = np.ravel(array_var5['n'])
print(np.max(array_var5))
#расчет минимального
print(np.min(array_var5))

#расчет медианы
print(np.median(array_var5))

#расчет среднего арифметического
print(np.mean(array_var5))

#расчет дисперсии
print(np.var(array_var5))

#расчет среднеквадратичного отклонения
print(np.std(array_var5))

plt.plot(array_var5)
#график
plt.show()
#значение массива, среднее значение и дисперсия на одном графике
mean = np.mean(array_var5) * np.ones(len(array_var5))
var = np.var(array_var5) * np.ones(len(array_var5))
plt.plot(array_var5, 'b-', mean, 'r-', mean - var, 'g--', mean + var, 'g--')
plt.grid()
plt.show()
plt.hist(array_var5, bins=20)
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

#одномерная матрица to одномерный массив затем выполняем корреляцию и строим график
cor = autocorrelate(array_var5)
plt.plot(cor)
plt.title("График 1")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

array_var2 = scipy.io.loadmat('./var2.mat')
array_var2.keys()
array_var2 = array_var2['mn']

n = array_var2.shape[1]
corr_matrix = np.zeros((n, n))
for i in range(0, n):
    for j in range(0, n):
        col = array_var2[:, i]
        col2 = array_var2[:, j]
        corr_matrix[i, j] = np.corrcoef(col, col2)[0, 1]
np.set_printoptions(precision=2)
print(corr_matrix)
plt.title("График 2")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(array_var2[:, 2], array_var2[:, 5], 'b.')
plt.grid()
plt.show()
