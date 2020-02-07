import os
from random import uniform

import numpy as np
from cv2.cv2 import imread
from numpy.linalg import norm

# Матрица эталонных образов
standard_images_matrix_1 = np.array([[-1, -1, -1],
                                     [-1, -1, 1],
                                     [-1, 1, -1],
                                     [-1, 1, 1],
                                     [1, -1, -1],
                                     [1, -1, 1],
                                     [1, 1, -1],
                                     [1, 1, 1]])

# Еще одна матрица
standard_images_matrix_2 = np.array([[-1, -1, 1, -1, 1, -1, 1, -1, 1],
                                     [-1, 1, -1, 1, 1, 1, -1, 1, -1],
                                     [1, 1, 1, 1, -1, 1, 1, 1, 1]])


class HammingNetwork:
    def __init__(self, images_matrix):
        # Число образов
        self.k = len(images_matrix)
        # Число признаков
        self.m = len(images_matrix[0])
        # Параметр пороговой функции
        self.t = self.m / 2
        # Матрица весовых коэффициентов нейронов первого слоя
        self.weights = np.zeros((self.k, self.m))
        for i in range(self.k):
            for j in range(self.m):
                self.weights[i][j] = images_matrix[i][j] / 2
        # Матрица весов обратных связей (весовых коэффициентов второго слоя)
        eps = uniform(0, 1 / self.k)
        self.e_layer = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                self.e_layer[i][j] = 1 if i == j else -eps
        self.e_max = 0.1

    # Пороговая функция активации
    def threshold_function(self, x):
        for i in range(len(x)):
            x[i] = {
                x[i] <= 0: 0,
                0 < x[i] <= self.t: x[i],
                x[i] > self.t: self.t
            }[True]
        return x

    def calculate_first_layer(self, x):
        result = self.weights @ x
        return self.threshold_function(result)

    def recalculate_second_layer(self, y):
        result = self.e_layer @ y
        return self.threshold_function(result)


def bmp_to_standard_matrix_row(path):
    result = np.intc(imread(path, 0).flatten())
    for i in range(len(result)):
        result[i] = 1 if (result[i] == 0) else -1
    return result


def main():
    # Чтение данных
    k = 10
    standard_matrix = []
    for i in range(k):
        img = bmp_to_standard_matrix_row(
            os.path.join(os.path.dirname(__file__), '..', 'resources\\images\\' + str(i) + '.bmp'))
        standard_matrix.append(img)
    standard_images_matrix = np.asarray(standard_matrix)

    # Обучение
    neuron = HammingNetwork(standard_images_matrix)
    # Использование
    x = bmp_to_standard_matrix_row(
        os.path.join(os.path.dirname(__file__), '..', 'resources\\images\\test.bmp'))
    # x = np.array([1, -1, -1, -1, 1, -1, 1, -1, 1])
    y = neuron.calculate_first_layer(x)
    y_next = neuron.recalculate_second_layer(y)

    while norm(y_next - y) ** 2 > neuron.e_max:
        y = y_next
        y_next = neuron.recalculate_second_layer(y)
    print(y_next)


if __name__ == '__main__':
    main()
