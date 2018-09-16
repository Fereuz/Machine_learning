# -*- coding: utf-8 -*-

import sys
import struct
import numpy as np
from scipy.special import expit


def load_mnist(kind = 'train'):
    """Загрузка данных MNIST"""
        
    with open('mnist/{}-labels.idx1-ubyte'.format(kind), 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)

    with open('mnist/{}-images.idx3-ubyte'.format(kind), 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype = np.uint8).reshape(len(labels), 784)
 
    return images, labels


# MLP - многослойный перцептрон
class NeuralNetMLP(object):
    """ Нейронная сеть с прямым распространением сигналов
    Классификатор на основе многослойного персептрона.

    Параметры
    ---------
    n_output : int
        Число выходных узлов.
        Должно равняться числу уникальных меток классов.

    n_features : int
        Число признаков (размерностей) в целевом наборе данных.
        Должно равняться числу столбцов в массиве Х.

    n_hidden : int (по умолчанию: 30)
        Число скрытых узлов.

    l1 : float (по умолчанию: 0.0)
        Значение лямбды для L1 - регуляризации.
        Регуляризация отсутствует, если l1 = 0.0 (по умолчанию)

    l2 : float (по умолчанию: 0.0)
        Значения лямбды для L2 - регуляризации.
        Регуляризация отсутствует, если l2 = 0.0 (по умолчанию)

    epochs : int (по умолчанию: 500)
        Число проходов по тренировочному набору.

    eta : float (по умолчанию: 0.001)
        Темп обучения.

    alpha : float (по умолчанию: 0.0)
        Константа импульса. Фактор, помноженный на градиент
        предыдущей эпохи t - 1, с целью улучшить скорость обучения
        w(t) := w(t) - (grad(t) + alpha*grad(t - 1))

    decrease_const : float (по умолчанию: 0.0)
        Константа уменьшения. Сокращает (стягивает) темп обучения
        после каждой эпохи посредством eta / (1 + эпоха * константа_уменьшения)

    shuffle : bool (по умолчанию: True)
        Перемешивает тренировочные данные в каждой эпохе
        для предотвращения зацикливания, если True.

    minibatches : int (по умолчанию: 1)
        Разбивает тренировочные данные на k мини-пакетов для эффективности.
        Обучение с нормальным градиентным спуском, если k = l (по умолчанию).

    random_state : int (по умолчанию: None)
        Инициализирует генератор случайных чисел
        для перемешивания и инициализации весов.

    Атрибуты
    --------
    cost_ : list/cпиcoк
        Сумма квадратичных ошибок после каждой эпохи.
"""

    def __init__(self, n_output, n_features, n_hidden = 30,
                 l1 = 0.0, l2 = 0.0, epochs = 500, eta = 0.001,
                 alpha = 0.0, decrease_const = 0.0, shuffle = True,
                 minibatches = 1, random_state = None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches


    def _encode_labels(self, y, k):
        """Преобразовать метки в прямую кодировку one-hot

        Параметры
        ---------
        y : массив, форма = [n_samples]
            Целые значения.

        Возвращает
        ----------
        onehot : массив, форма = (n_labels, n_samples)
        """

        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0

        return onehot


    def _initialize_weights(self):
        """Инициализировать веса малыми случайными числами"""

        w1 = np.random.uniform(-1.0, 1.0,
                               size = self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0,
                               size = self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2


    def _sigmoid(self, z):
        """Вычислить логистическую функцию (сигмоиду).
        Использует функцию scipy.special.expit, чтобы избежать ошибки
        переполения для очень мылых входных значений z.
        """

        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)


    def _sigmoid_gradient(self, z):
        """Вычислить градиент логистической функции"""

        sg = self._sigmoid(z)
        return sg * (1.0 - sg)


    def _add_bias_unit(self, X, how = 'column'):
        """Добавить в массив узел смещения (столбец или строку из единиц)
        в нулевом индексе"""

        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new


    def _feedforward(self, X, w1, w2):
        """Вычислить шаг прямого распространения сигнала

        Параметры
        ---------
        X : массив, форма = [n_samples, n_features]
            Входной слой с исходными признаками.
            
        w1 : массив, форма = [n_hidden_units, n_features]
            Матрица весовых коэффициентов для "входной слой -> скрытый слой".
            
        w2 : массив, форма = [n_output_units, n_hidden_units]
            Матрица весовых коэффициентов для "скрытый слой -> выходной слой".
            
        Возвращает
        ----------
        a1 : массив, форма = [n_samples, n_features + 1]
            Входные значения с узлом смещения.
            
        z2 : массив, форма = [n_hidden, n_samples]
            Чистый вход скрытого слоя.
            
        a2 : массив, форма = [n_hidden + 1, n_samples]
            Активация скрытого слоя.
            
        z3 : массив, форма = [n_output_units, n_samples]
            Чистый вход выходного слоя.
            
        a3 : массив, форма = [n_output_units, n_samples]
            Активация выходного слоя.
        """

        a1 = self._add_bias_unit(X, how = 'column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how = 'row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3


    def _L2_reg(self, lambda_, w1, w2):
        """Вычислить L2-регуляризованную стоимость"""
        
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) +
                                np.sum(w2[:, 1:] ** 2))


    def _L1_reg(self, lambda_, w1, w2):
        """Вычислить L1-регуляризованную стоимость"""

        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() +
                                np.abs(w2[:, 1:]).sum())


    def _get_cost(self, y_enc, output, w1, w2):
        """Вычислить функцию стоимости.

        Параметры
        ---------
        y_enc : массив, форма = (n_labels, n_samples)
            Прямокодированные метки классов.
            
        output : массив, форма = [n_output_units, n_samples]
            Активация выходного слоя (прямое распространение).
            
        w1 : массив, форма = [n_hidden_units, n_features]
            Матрица весовых коэффициентов для "входной слой -> скрытый слой".
            
        w2 : массив, форма = [n_output_units, n_hidden_units]
            Матрица весовых коэффициентов для "скрытый слой -> выходной слой".
            
        Возвращает
        ----------
        cost : float
            Регуляризованная стоимость.
        """
        
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost


    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """Вычислить шаг градиента, используя обратное распространение.

        Параметры
        ---------
        a1 : массив, форма = [n_samples, n_features + 1]
            Входные значения с узлом смещения.
            
        a2 : массив, форма = [n_hidden + 1, n_samples]
            Активация скрытого слоя.
            
        a3 : массив, форма = [n_output_units, n_samples]
            Активация выходного слоя.
            
        z2 : массив, форма = [n_hidden, n_samples]
            Чистый вход скрытого слоя.
            
        y_enc : массив, форма = (n_labels, n_samples)
            Прямокодированные метки классов.
            
        w1 : массив, форма = [n_hidden_units, n_features]
            Матрица весовых коэффициентов для "входной слой -> скрытый слой".
            
        w2 : массив, форма = [n_output_units, n_hidden_units]
            Матрица весовых коэффициентов для "скрытый слой -> выходной слой".
            
        Возвращает
        ----------
        grad1 : массив, форма = [n_hidden_units, n_features]
            Градиент матрицы весовых коэффициентов w1.
            
        grad2 : массив, форма = [n_output_units, n_hidden_units]
            Градиент матрицы весовых коэффициентов w2.
        """

        # Обратное распространение
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how = 'row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # Регуляризовать
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2


    def predict(self, X):
        """Предсказать метки классов.

        Параметры
        ---------
        X : массив, форма = [n_samples, n_features]
            Входной слой с исходными признаками.
            
        Возвращает
        ----------
        y_pred : массив, форма = [n_samples]
            Предсказанные метки классов.
        """
        
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred


    def fit(self, X, y, print_progress = False):
        """Извлечь веса из тренировочных данных.

        Параметры
        ---------
        X : массив, форма = [n_samples, n_features]
            Входной слой с исходными признаками.
            
        y : массив, форма = [n_samples]
            Целевые метки классво.
            
        print_progress : bool (default: False)
            Распечатывать ход работы в виде числа эпох на устройстве stderr.
            
        Возвращает
        ----------
        self
        """
        
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            # Адаптивный темп обучения
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rЭпоха: %d/%d \n' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:

                # Прямое распространение
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx],
                                                       self.w1,
                                                       self.w2)
                cost = self._get_cost(y_enc = y_enc[:, idx],
                                      output = a3,
                                      w1 = self.w1,
                                      w2 = self.w2)
                self.cost_.append(cost)

                # Вычислить градиент методом обратного распространения
                grad1, grad2 = self._get_gradient(a1 = a1, a2 = a2,
                                                  a3 = a3, z2 = z2,
                                                  y_enc = y_enc[:, idx],
                                                  w1 = self.w1,
                                                  w2 = self.w2)

                # Обновить веса
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self


#Загружаем тренировочные экземпляры
X_train, y_train = load_mnist(kind = 'train')
print('Тренировка - строки: {}, столбцы: {}'.format(X_train.shape[0],
                                                    X_train.shape[1]))

#Загружаем тестовые образцы
X_test, y_test = load_mnist(kind = 't10k')
print('Тестирование - строки: {}, столбцы: {}'.format(X_test.shape[0],
                                                      X_test.shape[1]))

"""
#Посмотреть, как выглядят примеры цифр
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
ax = ax.flatten()

for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('./figures/mnist_all.png', dpi=300)
plt.show()

#Посмотреть, как отличаются почерки
fig, ax = plt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 0][i].reshape(28, 28)
    ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('./figures/mnist_7.png', dpi=300)
plt.show()
"""


#Инициализация нового многослойного перцептрона с конфигурацией 784-50-10
#784 - входных узлов (n_features), 50 - скрытых узлов (n_hidden),
#10 - выходных узлов (n_output)
nn = NeuralNetMLP(n_output = 10, 
                  n_features = X_train.shape[1], 
                  n_hidden = 50, 
                  l2 = 0.1, 
                  l1 = 0.0, 
                  epochs = 1000, 
                  eta = 0.001,
                  alpha = 0.001,
                  decrease_const = 0.00001,
                  minibatches = 50, 
                  shuffle = True,
                  random_state = 1)

#Тренируем MLP
nn.fit(X_train, y_train, print_progress = True)
