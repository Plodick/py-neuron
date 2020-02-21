from tensorflow.keras import Sequential
from tensorflow.keras import utils
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense


def main():
    # Загружаем данные
    (x_training, y_training), (x_test, y_test) = fashion_mnist.load_data()

    # Список с названиями классов
    classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто',
               'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

    # Преобразование размерности изображений
    x_training = x_training.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Нормализация данных
    x_training = x_training / 255
    x_test = x_test / 255

    # Преобразуем метки в категории
    y_training = utils.to_categorical(y_training, 10)
    y_test = utils.to_categorical(y_test, 10)

    # Создаем последовательную модель
    model = Sequential()

    # Добавляем уровни сети
    model.add(Dense(800, input_dim=784, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # Компилируем модель
    model.compile(loss="categorical_crossentropy",
                  optimizer="SGD",
                  metrics=["accuracy"])

    print(model.summary())

    # Обучаем сеть
    history = model.fit(x_training, y_training,
                        batch_size=200,
                        epochs=100,
                        validation_split=0.2,
                        verbose=1)

    # Оцениваем качество обучения сети на тестовых данных
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Доля верных ответов на тестовых данных, в процентах:",
          round(scores[1] * 100, 4))


if __name__ == '__main__':
    main()
