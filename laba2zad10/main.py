import numpy as np

print('Введите размер массива: \t')

try:
    x = int(input())
except:
    print("введите целое число")


def generate(n):
    a = np.random.randint(0, 100, (n))
    res1, res2 = min(a), max(a)
    return res1, res2


result = generate(x)
print(result)
