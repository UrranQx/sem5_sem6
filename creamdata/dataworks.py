import numpy as np

# Входные данные: 10 положительных целых чисел
# numbers = [52, 69, 69, 22, 10, 43, 34, 82, 72, 64]
numbers = [19, 73, 68, 94, 34, 33, 76, 69, 37, 71]
# Шаг 1: Вычисляем экспоненты
exp_values = np.exp(numbers)

# Шаг 2: Находим сумму экспонент
sum_exp_values = np.sum(exp_values)

# Шаг 3: Нормируем значения
normalized_values = exp_values / sum_exp_values

# Округляем до тысячных


# Выводим результат
for value in normalized_values:
    print(f"{value:.3f}")  # Округление до 5 знаков после запятой

print('real shit')


def y(x, min):
    return 1 - np.exp(1 - (x / min))


min = min(numbers)
new_values = [y(x, min) for x in numbers]
for i in range(len(new_values)):
    print(f'exp_norm: {new_values[i]:.4f}\t\tокруглить до 3 знаков\t\t{new_values[i]:.3f}:\tbefore {numbers[i]}')
