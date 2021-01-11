import math
import time


def fibonacci(n):
    if n in (1, 2):
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(10))


def fac(n):
    if n == 0:
        return 1
    return fac(n - 1) * n


print('Введите число: \t')
x = int(input())
start_time = time.time()
print(fac(x))
print("%s seconds" % (time.time() - start_time))

start_time2 = time.time()
print(math.factorial(x))
print("%s seconds" % (time.time() - start_time2))
