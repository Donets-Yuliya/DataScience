#data_loading.py (модуль загрузки файла csv)
import pandas as pd
import os

def is_file_exist(file_path):
  #Проверка наличия файла
  if os.path.isfile(file_path):
    print(f"Файл {file_path} найден")
    return True
  else:
    print(f"Файл {file_path} отсутствует")
    return False

def is_file_empty(file_path):
    #Проверка пустой ли файл (True, если пустой или есть ошибка)
    try:
      loaded_file = open(file_path, 'r')
      first_string = loaded_file.read()

      if len(first_string) == 0:
          print("Файл пуст")
          return True
      else:
          print("Файл не пустой")
          return False
    except IOError as e:
        print(f"Ошибка чтения файла: {e}")
        return True

def load_data(file_path):
  
    #Проверка наличия файла
    if is_file_exist(file_path) and not is_file_empty(file_path):
      print("Файл успешно загружен")
      return pd.read_csv(file_path)
    else:
      print("Файл не загружен")
