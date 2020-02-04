from random import uniform

import numpy as np

# Матрица эталонных образов
standard_images_matrix = np.array([[-1, -1, -1],
                                   [-1, -1, 1],
                                   [-1, 1, -1],
                                   [-1, 1, 1],
                                   [1, -1, -1],
                                   [1, -1, 1],
                                   [1, 1, -1],
                                   [1, 1, 1]])

# Параметр пороговой функции
T = len(standard_images_matrix[0]) / 2


# Пороговая функция активации
def threshold_function(x):
    return {
        x <= 0: 0,
        0 < x <= T: x,
        x > T: T
    }[True]


class HammingNeuron:
    def __init__(self, images_matrix):
        # Число образов
        self.k = len(images_matrix)
        # Число признаков
        self.m = len(images_matrix[0])
        # Матрица весовых коэффициентов нейронов первого слоя
        self.weights = np.zeros((self.k, self.m))
        for i in range(self.k):
            for j in range(self.m):
                self.weights[i][j] = images_matrix[i][j] / 2
        # Задание обратных нейронов второго слоя
        eps = uniform(0, 1 / self.k)
        self.eps_matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                self.eps_matrix[i][j] = 1 if i == j else -eps
        self.e_max = 0.1


def main():
    # Обучение
    neuron = HammingNeuron(standard_images_matrix)
    # Использование

    print()


if __name__ == '__main__':
    main()
