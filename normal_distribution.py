import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_normal_distribution(mean, std):
    x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    pdf = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, pdf, label=f'Mean={mean}, Std={std}')


params = [(10, 2), (10, 1), (10, 1 / 4), (12, 1)]

plt.figure(figsize=(10, 6))

for param in params:
    mean, std = param
    plot_normal_distribution(mean, std)

plt.title('Плотность вероятности нормального распределения')
plt.xlabel('Значение')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.show()

# Параметры распределения
mean, std = 10, 2

# Создание графика функции распределения
x = np.linspace(0, 20, 1000)
cdf = norm.cdf(x, loc=mean, scale=std)
plt.figure(figsize=(10, 6))
plt.plot(x, cdf)
plt.title('Функция распределения нормального распределения')
plt.xlabel('Значение')
plt.ylabel('Функция распределения')
plt.show()

# Параметры распределения
mean, std = 10, 2
N_values = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]  # Задайте нужные значения N

random_numbers = []

for N in N_values:
    uniform_samples = np.random.rand(N)  # Генерация равномерно распределенных чисел
    normal_samples = norm.ppf(uniform_samples, loc=mean, scale=std)  # Используем обратную функцию
    random_numbers.append(normal_samples)

# Графики можно построить, если нужно
for i in range(len(N_values)):
    plt.figure(figsize=(10, 6))
    plt.hist(random_numbers[i], bins=20, density=True, alpha=0.6)
    plt.title(f'Моделирование для N={N_values[i]}')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.show()

for i in range(len(N_values)):
    theoretical_distribution = norm.pdf(x, loc=mean, scale=std)
    plt.figure(figsize=(10, 6))
    plt.hist(random_numbers[i], bins=20, density=True, alpha=0.6, label='Экспериментальное распределение')
    plt.plot(x, theoretical_distribution, label='Теоретическое распределение')
    plt.title(f'N={N_values[i]}')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.show()

    # Расчет средней квадратичной погрешности
    experimental_pdf = np.histogram(random_numbers[i], bins=20, density=True)[0]
    mse = np.mean((theoretical_distribution[:20] - experimental_pdf) ** 2)
    print(f'N={N_values[i]}, MSE={mse}')
