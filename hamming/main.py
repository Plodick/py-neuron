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


class HammingNeuron:
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
        self.eps_matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                self.eps_matrix[i][j] = 1 if i == j else -eps
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

    def feed_forward(self, x):
        result = np.dot(self.weights, x) + self.t
        return self.threshold_function(result)

    def recalculate_second_layer(self, y):
        result = np.dot(self.eps_matrix, y)
        return self.threshold_function(result)


def bmp_to_standard_matrix_row(img):
    result = np.intc(img.flatten())
    for i in range(len(result)):
        result[i] = 1 if (result[i] == 0) else -1
    return result


def main():
    # Чтение данных
    k = 2
    standard_matrix = []
    for i in range(k):
        img = imread(os.path.join(os.path.dirname(__file__), '..', 'resources\\images\\2' + str(i) + '.bmp'), 0)
        standard_matrix.append(bmp_to_standard_matrix_row(img))
    standard_images_matrix = np.asarray(standard_matrix)

    # Обучение
    neuron = HammingNeuron(standard_images_matrix)
    # Использование
    x = bmp_to_standard_matrix_row(
        imread(os.path.join(os.path.dirname(__file__), '..', 'resources\\images\\test2.bmp'), 0))
    y = neuron.feed_forward(x)
    y_next = neuron.recalculate_second_layer(y)

    while norm(y_next - y) ** 2 > neuron.e_max:
        y = y_next
        y_next = neuron.recalculate_second_layer(y)
    print(y_next)


if __name__ == '__main__':
    main()
