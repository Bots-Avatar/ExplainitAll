import copy
import math
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io


def df_to_heatmap_plot(df, title=None):
    df_temp = copy.deepcopy(df)
    df_temp.set_index('Tokens', inplace=True)
    
    rows_number = df.shape[0]
    cols_number = df.shape[1]

    plt.figure(figsize=(max(10, math.floor(cols_number * 2.5)), max(10, math.floor(rows_number * 0.4))))
    sns.heatmap(df_temp, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='lightgrey')
    if title:
        plt.title(title)
    plt.yticks(rotation=45, va='top')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    #im.show(title="My Image")

    #img_buf.close()
    return im
