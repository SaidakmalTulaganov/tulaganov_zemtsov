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
# импорт данных из файла и указываем что там целые числа
print(data)

# загрузка одномерных данных
one_dimensional = scipy.io.loadmat('./var5.mat')
one_dimensional = np.ravel(one_dimensional['n'])
print(np.max(one_dimensional))
# расчет минимального
print(np.min(one_dimensional))

# расчет медианы
print(np.median(one_dimensional))

# расчет среднего арифметического
print(np.mean(one_dimensional))

# расчет дисперсии
print(np.var(one_dimensional))

# расчет среднеквадратичного отклонения
print(np.std(one_dimensional))

plt.plot(one_dimensional)
# график
plt.show()
# значение массива, среднее значение и дисперсия на одном графике
mean = np.mean(one_dimensional) * np.ones(len(one_dimensional))
var = np.var(one_dimensional) * np.ones(len(one_dimensional))
plt.plot(one_dimensional, 'b-', mean, 'r-', mean - var, 'g--', mean + var, 'g--')
plt.grid()
plt.show()
plt.hist(one_dimensional, bins=20)
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


# одномерная матрица to одномерный массив затем выполняем корреляцию и строим график
cor = autocorrelate(one_dimensional)
plt.plot(cor)
plt.title("График автокорреляции")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

multidimensional = (scipy.io.loadmat('./var2.mat'))['mn']

n = multidimensional.shape[1]
corr_matrix = np.zeros((n, n))
for i in range(0, n):
    for j in range(0, n):
        col = multidimensional[:, i]
        col2 = multidimensional[:, j]
        corr_matrix[i, j] = np.corrcoef(col, col2)[0, 1]
np.set_printoptions(precision=2)
print(corr_matrix)
plt.title("Точечный график автокорреляции двух столбцов")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(multidimensional[:, 2], multidimensional[:, 5], 'b.')
plt.grid()
plt.show()
