import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

from explainitall import stat_helpers


def show_distribution_histogram(data):
    """
    Строит гистограмму распределения
    """
    stat_params = stat_helpers.calc_gmm_stat_params(data)
    data = data[~np.isnan(data)]
    data.sort()

    gaussian_prob = stat_helpers.compute_gaussian_integral(data, stat_params["mean"], stat_params["std"])
    rayleigh_prob = stat_helpers.rayleigh_integral(data, data.var())

    plt.hist(data, bins=15, cumulative=True, density=True, label='Распределение реальных данных')
    plt.plot(data, gaussian_prob, color='red', label='Функция Гаусса')
    plt.plot(data, rayleigh_prob, label='Функция Рэлея')

    plt.legend(title='Функции распределения')

    ax = plt.gca()
    ax.set_xlabel("Значения", fontsize=10, color='black', labelpad=10)
    ax.set_ylabel("Распределение  P(X<x)", fontsize=10, color='black', labelpad=10)
    ax.grid()
    plt.show()


def show_distribution_plot(array: np.ndarray):
    """
    Строит график распределения
    """
    dic_stat = stat_helpers.calc_gmm_stat_params(array)
    arr = array[~np.isnan(array)]
    arr.sort()
    qqplot((arr - dic_stat["mean"]) / dic_stat["std"], line='45')
    plt.grid()
    plt.show()


def df_to_heatmap(df, title=None, annot=True):
    try:
        df_temp = copy.deepcopy(df)
        df_temp.set_index('Tokens', inplace=True)

        plt.figure(figsize=(15, 10))
        sns.heatmap(df_temp, annot=annot, cmap='coolwarm', linewidths=0.5, linecolor='lightgrey')
        if title:
            plt.title(title, fontsize=14)
        plt.yticks(rotation=45, va='top', fontsize=10)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.show()
    except:
        print("Данные отсутствуют или неполны.")
        plt.figure(figsize=(15, 10))
        plt.text(0.5, 0.5, 'Данные отсутствуют или неполны, Нет данных для отображения', fontsize=14, ha='center')
