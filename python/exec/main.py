import numpy as np

T = 1

standart_images_matrix = np.array([[-1, -1, -1],
                                   [-1, -1, 1],
                                   [-1, 1, -1],
                                   [-1, 1, 1],
                                   [1, -1, -1],
                                   [1, -1, 1],
                                   [1, 1, -1],
                                   [1, 1, 1]])


def threshold_function(x):
    return {
        x <= 0: 0,
        0 < x <= T: x,
        x > T: T
    }[True]


class HammingNeuron:
    def __init__(self, images_matrix):
        self.weights = np.zeros((len(images_matrix), len(images_matrix[0])))
        for i in range(len(images_matrix)):
            for j in range(len(images_matrix[i])):
                self.weights[i][j] = images_matrix[i][j] / 2


def main():
    neuron = HammingNeuron(standart_images_matrix)
    print()


if __name__ == '__main__':
    main()
