import copy
import io
import math

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def df_to_heatmap_plot(df, title=None):
    try:
        df_temp = copy.deepcopy(df)
        df_temp.set_index('Tokens', inplace=True)

        rows_number = df.shape[0]
        cols_number = df.shape[1]

        plt.figure(figsize=(max(10, math.floor(cols_number * 2.5)), max(10, math.floor(rows_number * 0.4))))
        sns.heatmap(df_temp, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='lightgrey')
        if title:
            plt.title(title)
        plt.yticks(rotation=45, va='top')
    except:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Нет данных для отображения', fontsize=14, ha='center')
        plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    return Image.open(img_buf)
