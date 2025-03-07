import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as f_activation

DIRECTORY = 'dataset/'
TRAIN_FILENAME = 'mnist_train.csv'
TEST_FILENAME = 'mnist_test.csv'


# The expit function, also known as the logistic sigmoid function, is defined as expit(x) = 1/(1+exp(-x)).
# It is the inverse of the logit function.

def init_net():
    input_nodes = 28 * 28  # 784 # размер изображения из базы данных MNIST
    print('Input the number of hidden neurons: ')
    hidden_nodes = int(input())
    out_nodes = 10  # Классификация цифр от 0 до 9
    print('Input the training speed (0.5): ')
    learn_speed = float(input())

    return input_nodes, hidden_nodes, out_nodes, learn_speed


def create_net(input_nodes, hidden_nodes, out_nodes):
    # Инициализация весов рандомными значениями
    # Для корректного умножения правый элемент в размерности должен совпадать с размерностью того, на что мы умножаем
    weights_in2hidden = np.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
    weights_hidden2out = np.random.uniform(-0.5, 0.5, (out_nodes, hidden_nodes))

    return weights_in2hidden, weights_hidden2out


def net_output(weights_in2hidden, weights_hidden2out, input_signal, return_hidden=1):
    inputs = np.array(input_signal, ndmin=2).T

    hidden_in = weights_in2hidden @ inputs
    hidden_out = f_activation(hidden_in)

    final_in = weights_hidden2out @ hidden_out
    final_out = f_activation(final_in)

    if return_hidden:
        return final_out, hidden_out

    return final_out


def net_train(target_list, input_signal, weights_in2hidden, weights_hidden2out, learn_speed):
    targets = np.array(target_list, ndmin=2).T
    inputs = np.array(input_signal, ndmin=2).T
    final_out, hidden_out = net_output(weights_in2hidden, weights_hidden2out, input_signal)

    out_errors = targets - final_out  # Ошибки выходного слоя
    hidden_errors = weights_hidden2out.T @ out_errors  # Ошибки скрытого слоя

    weights_hidden2out += learn_speed * (out_errors * final_out * (1 - final_out)) @ hidden_out.T
    weights_in2hidden += learn_speed * (hidden_errors * hidden_out * (1 - hidden_out)) @ inputs.T

    return weights_in2hidden, weights_hidden2out


def train_set(weights_in2hidden, weights_hidden2out, learn_speed):
    with open(DIRECTORY + TRAIN_FILENAME, 'r') as file:
        training_list = file.readlines()  # Все данные сразу загружаются в переменную (Плохо если объем данных большой)
        file.close()

    for record in training_list:
        all_values = record.split(',')
        inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255 * 0.999) + 0.001  # нормализация данных
        targets = np.zeros(10) + 0.001
        targets[int(all_values[0])] = 1  # В массиве targets все нули + 0.001 кроме правильного ответа
        weights_in2hidden, weights_hidden2out = net_train(targets, inputs, weights_in2hidden, weights_hidden2out,
                                                          learn_speed)

    return weights_in2hidden, weights_hidden2out


def test_set(weights_in2hidden, weights_hidden2out):
    with open(DIRECTORY + TEST_FILENAME, 'r') as file:
        test_list = file.readlines()
        file.close()

    test = []
    for record in test_list:
        all_values = record.split(',')
        inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255 * 0.999) + 0.001
        out_session = net_output(weights_in2hidden, weights_hidden2out, inputs, return_hidden=0)

        if int(all_values[0]) == np.argmax(out_session):
            test.append(1)
        else:
            test.append(0)

    test = np.asarray(test)
    print(f'Net efficiency % = {test.sum() / test.size * 100}')
    return test


def plot_image(pixels: np.array):
    plt.imshow(pixels.reshape((28, 28)), cmap='gray')
    plt.show()


input_nodes, hidden_nodes, out_nodes, learn_speed = init_net()

weights_in2hidden, weights_hidden2out = create_net(input_nodes, hidden_nodes, out_nodes)
NUM_ITERATIONS = 10
for i in range(NUM_ITERATIONS):
    print('Test #', i + 1)
    weights_in2hidden, weights_hidden2out = train_set(weights_in2hidden, weights_hidden2out, learn_speed)
    test_set(weights_in2hidden, weights_hidden2out)

with open(DIRECTORY + TEST_FILENAME, 'r') as data_file:
    test_data = data_file.readlines()
    data_file.close()
    values = test_data[int(np.random.uniform(0, 9999))].split(',')
    inputs = (np.asarray(values[1:], dtype=np.float64) / 255 * 0.999) + 0.001
    out_session = net_output(weights_in2hidden, weights_hidden2out, inputs, return_hidden=0)
    print(np.argmax(out_session))
    plot_image(np.asarray(values[1:], dtype=np.float64))
