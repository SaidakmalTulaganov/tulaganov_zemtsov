import numpy as np

print('Введите размер массива: \t')
x = int(input())


def generate(n):
    a = np.random.randint(0, 100, (n))
    res = (min(a), max(a))
    print(res)


generate(x)
