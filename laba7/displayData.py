import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt


def displayData(X, title=''):

    # Вычислить строки, столбцы
    m, n = X.shape
    example_width = int(round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Вычислить количество отображаемых элементов
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Заполнение между изображениями
    pad = 1

    # Настройка пустого дисплея
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                           pad + display_cols * (example_width + pad)))

    # Скопируйте каждый пример в патч на массиве дисплея
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, : ]))
            rows = [pad + j * (example_height + pad) + x for x in np.arange(example_height+1)]
            cols = [pad + i * (example_width + pad)  + x for x in np.arange(example_width+1)]
            display_array[min(rows):max(rows), min(cols):max(cols)] = X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

    # Показать изображение
    display_array = display_array.astype('float32')
    plt.imshow(display_array.T)
    plt.set_cmap('gray')
    plt.title(title)

    # Не показывать ось
    plt.axis('off')
    plt.show()

