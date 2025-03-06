from pulp import *

# Создаем модель
prob = LpProblem("Production Planning", LpMaximize)

# Переменные решения
x = LpVariable.dicts("x", range(1, 6), lowBound=0, cat='Integer')
y = LpVariable.dicts("y", range(1, 6), lowBound=0, cat='Integer')
K = LpVariable("K", lowBound=0)

# Целевая функция
prob += K

# Ограничения
prob += K <= (100 * x[1] + 400 * x[2] + 20 * x[3] + 200 * x[4] + 600 * x[5]) / 2
prob += K <= 15 * y[1] + 200 * y[2] + 2.5 * y[3] + 50 * y[4] + 250 * y[5]

prob += x[1] + y[1] <= 5
prob += x[2] + y[2] <= 3
prob += x[3] + y[3] <= 40
prob += x[4] + y[4] <= 9
prob += x[5] + y[5] <= 2

# Решаем задачу
prob.solve()

# Выводим результаты
print("Статус:", LpStatus[prob.status])
print("Максимальное количество комплектов (тыс.):", value(prob.objective))
print("\nРаспределение предприятий:")
for i in range(1, 6):
    if value(x[i]) > 0:
        print(f"Тип {i}, Изделие 1: {int(value(x[i]))} предприятий")
    if value(y[i]) > 0:
        print(f"Тип {i}, Изделие 2: {int(value(y[i]))} предприятий")