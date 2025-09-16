#data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def analise_nulls(df):
  #Проверка на пропущенные значения
  null_columns = df.columns[df.isnull().any()]

  if null_columns.empty:
    print("Пропущенные значения отсутствуют")
  else:
    print(f"Пропущенные значения обнаружены в {null_columns.array.size} столбцах")

def show_nulls(df):
  null_columns = df.columns[df.isnull().any()]

  if null_columns.empty:
    print("Пропущенные значения отсутствуют")
  else:
    print(df[null_columns].isnull().sum())
    
def fill_nulls(df):
  null_columns = df.columns[df.isnull().any()]

  if null_columns.empty:
    print("Пропущенные значения отсутствуют, заполнение не выполнено")
  else:
    for column in null_columns.array:
      df[column] = df[column].fillna(df[column].mode()[0]) #заполнение наиболее часто встречающимися значениями
    
    print("Заполнение пропущенных значений выполнено успешно")

  return df

def preprocess_data(df, numeric_features, target_column):
  #Предобработка данных: разделение на признаки и целевую переменную, масштабирование признаков.
  #return: Обработанные признаки, целевая переменная, препроцессор.
  X = df.drop(columns=[target_column])
  y = df[target_column]

  # Создание препроцессора
  numeric_transformer = StandardScaler()

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', numeric_transformer, numeric_features)
      ])

  # Применение препроцессора к данным
  X_processed = preprocessor.fit_transform(X)
  
  return X_processed, y, preprocessor

def preprocess_data(df, numeric_features, categorical_features, target_column):
  #Предобработка данных: разделение на признаки и целевую переменную, масштабирование признаков.
  #return: Обработанные признаки, целевая переменная, препроцессор.
  X = df.drop(columns=[target_column])
  y = df[target_column]

  # Создание препроцессора
  numeric_transformer = StandardScaler()
  categorical_transformer = OneHotEncoder(drop='first')
  
  if categorical_features.count == 0:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
  else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

  # Применение препроцессора к данным
  X_processed = preprocessor.fit_transform(X)
  
  return X_processed, y, preprocessor