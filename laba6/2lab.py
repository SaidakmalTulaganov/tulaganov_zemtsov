import matplotlib.pyplot as plt
import numpy as np

#1 задание
datasave = np.matrix(np.loadtxt('ex1data1.txt', delimiter=','))
def compute_cost(x, y ,theta):
    m = x.shape[0] #количество элементов в X(количество городов)
    x_ones = np.c_[np.ones((m, 1)), x] #добавляем единичный столбец к Х
    h_x = x_ones * theta #вычисление гипотезы для вех городов сразу
    cost = sum(np.power(h_x - y, 2))/(2*x.shape[0])
    return cost
def gradientDescent(x, y, alpha, iterations):
    xs = []
    ys = []
    x = np.array(x)
    y = y.view(np.ndarray)
    y.shape = -1
    m=len(x)
    x=np.c_[np.ones((m, 1)),x]
    n = x.shape[1]
    xTrans = x.transpose()
    theta=np.ones(n)
    for i in range(0, iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        xs.append(i)
        ys.append(cost)
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    plt.title('Снижение ошибки при градиентном спуске')
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка')
    plt.xlim(0, iterations)
    plt.plot(xs, ys)
    plt.show()
    return theta
def minsquare(x , y):
    m = len(x)
    x=np.c_[np.ones((m, 1)),x]
    theta = np.linalg.pinv(x.transpose()*x)*x.transpose()*y
    return theta
#2 задание
x = []
y = []
from matplotlib import rc
font = {'family': 'Verdana', 'weight' : 'normal'}
rc('font', **font)
x = datasave[: , 0]
y = datasave[: , 1]
plt.plot(x, y, 'b.')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.title('Зависимость прибыльности от численности')
plt.grid()
plt.xlim(0, 25)
plt.ylim(-5, 25)
plt.show()
theta = np.matrix('[1; 2]')

#3 задание
print("\n\nКвадратичная ошибка:", compute_cost(x, y, theta)[0, 0])
#4 задание
alpha = 0.02
iteration  = 500
theta = gradientDescent(x, y, alpha, iteration)
print(theta)
population  = 50
print("Численность в 10 тыс. человек:",population ,"\nПрибыльность в 10.000 кратном размере: " ,theta[1]*population + theta[0] )
X_1 = np.arange(min(x), max(x))
#6 задание
plt.plot(X_1, theta[1]*X_1 + theta[0], 'g--')
plt.plot(x, y, 'b.')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.title('График линейной зависимости')
plt.xlim(0, 25)
plt.ylim(-5, 25)
plt.grid()
plt.show()
print("Фактическая цена: ", y[9],"\nПредсказанная цена: ",theta[1]*x[9] + theta[0])

#Многомерная линейная регрессия по методу градиентного спуска
#7 задание
datasave_2 = np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))
print('\n Многомерная линейная регрессия по методу градиентного спуска \n')
X_2 = datasave_2[:, :2]
Y_2 = datasave_2[:, 2]
mean_1 = X_2.mean(axis = 0)
std_1 = X_2.std(axis = 0)
X_21 = X_2 - mean_1
X_21 = X_21 / std_1
mean_2 = Y_2.mean(axis = 0)
std_2 = Y_2.std(axis = 0)
Y_21 = Y_2 - mean_2
Y_21 = Y_21 / std_2
#8 задание
Area = 3000
Count = 3
area = (Area - mean_1[0, 0])/std_1[0, 0]
count = (Count - mean_1[0, 1])/std_1[0, 1]
# print(count, area)
theta = gradientDescent(X_21, Y_21, 0.01, 500)
print("Площадь: ",Area ,"\nКоличество комнат:", Count,"\nПредсказанная цена: ", ((theta[2]*count + theta[1]*area+ theta[0])*std_2 + mean_2)[0, 0])
X_31 = np.arange(min(X_21[:, 0]), max(X_21[:, 0]))
# plt.plot(X_31, theta[1]*X_31 +  theta[2], 'g--')
# plt.plot(X_21[:, 0], Y_21, 'b.')
# plt.xlabel('Размер квартиры')
# plt.ylabel('Стоимость')
# plt.title('График линейной зависимости')
# plt.grid()
# plt.show()
X_32 = np.arange(min(X_21[:, 1]), max(X_21[:, 1]))
# plt.plot(X_32, theta[1]*X_32 +  theta[2], 'r--')
# plt.plot(X_21[:, 1], Y_21, 'y.')
# plt.xlabel('Количество комнат')
# plt.ylabel('Стоимость')
# plt.title('График линейной зависимости')
# plt.grid()
# plt.show()
#Метод наименьших квадратов
#9 задание
print('\n Метод наименьших квадратов \n')
theta_2 = minsquare(X_2, Y_2)
theta_2 = theta_2.view(np.ndarray)
theta_2.shape = -1
X_33 = np.arange(min(X_2[:, 1]), max(X_2[:, 1]))
# plt.plot(X_33, theta_2[2]*X_33 +  theta_2[0], 'r--')
# plt.plot(X_2[:, 1], Y_2, 'y.')
# plt.xlabel('Количество комнат')
# plt.ylabel('Стоимость')
# plt.title('График линейной зависимости')
# plt.grid()
# plt.show()
X_34 = np.arange(min(X_2[:, 0]), max(X_2[:, 0]))
# plt.plot(X_34, theta_2[1]*X_34 +  theta_2[0], 'g--')
# plt.plot(X_2[:, 0], Y_2, 'b.')
# plt.xlabel('Размер квартиры')
# plt.ylabel('Стоимость')
# plt.title('График линейной зависимости')
# plt.grid()
# plt.show()
print("Площадь: ",Area ,"\nКоличество комнат:", Count,"\nПредсказанная цена: ", (theta_2[1]*Area + theta_2[2]*Count+ theta_2[0]))
#Сравнение двух методов
#10 задание
print("\n\n\n\nМетод градиентного спуска:\nФактическая цена: ", (Y_21[1]*std_2 + mean_2)[0, 0],"\nПредсказанная цена: ", ((theta[1]*X_21[1, 0] + theta[2]*X_21[1, 1]+ theta[0])*std_2 + mean_2)[0, 0])
print("\nМетод наименьших квадтратов:\nФактическая цена: ", (Y_2[1, 0]),"\nПредсказанная цена: ", (theta_2[1]*X_2[1, 0] + theta_2[2]*X_2[1, 1]+ theta_2[0]))
