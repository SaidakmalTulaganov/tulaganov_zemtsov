import numpy as np

array1 = np.random.randint(0, 20, 10)
array2 = np.random.randint(0, 20, 10)


def print_hi(arr1, arr2):
    arr = [abs(arr1[j] - arr2[j]) for j in range(10)]
    [print(arr[j]) for j in range(len(arr))]


print_hi(array1, array2)
