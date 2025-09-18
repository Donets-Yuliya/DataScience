import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def pairplot_visualization(df, target):
  #Построение парных графиков
  
  if (target in df.columns):
    sns.pairplot(df, hue=target)
    plt.show()
  else:
    print(f"В датафрейме отсутствует поле {target}")

def heatmap_visualization(df):
  #Построение корреляционной матрицы
  numeric_df = df.select_dtypes(include=[np.number])  # Исключаем нечисловые столбцы
  corr = numeric_df.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  plt.figure(figsize=(18, 12))
  plt.title('Тепловая карта корреляций', fontsize=20)

  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.4, center=-0.4,
              square=True, linewidths=.5, cbar_kws={"shrink": 0.7}, annot=True)
  plt.show()

def boxplot_visualization(df):
  #Построение ящиков с усами
  sns.set(style="whitegrid")
  plt.figure(figsize=(12, 10))

  # Перебираем каждый числовой столбец и создаем для него ящик с усами
  for index, column in enumerate(df.select_dtypes(include=[np.number]).columns):
    plt.subplot((len(df.columns) // 3) + 1, 3, index + 1)
    sns.boxplot(y=df[column])

  plt.tight_layout()
  plt.show()

def hist_visualization(df):
  sns.set(style="whitegrid")

  # Создание гистограмм для каждой числовой переменной
  df.hist(bins=10, figsize=(10, 6), color='skyblue', edgecolor='black')

  # Добавление названий для каждого графика и осей
  for ax in plt.gcf().get_axes():
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
 
  # Регулировка макета для предотвращения наложения подписей
  plt.tight_layout()
  plt.show()
