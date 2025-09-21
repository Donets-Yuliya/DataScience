#train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def train(X, y):

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def predict(model, X):

    return model.predict(X)

def evaluate(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, report, confusion

# Визуализация истинных и предсказанных значений
def plot_predictions(confusion, class_names):
    #Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.grid(False)
    
    # Рисуем новые линии сетки на сдвинутых позициях
    for i in tick_marks:
      plt.axvline(x = i + 0.5, color='gray', linestyle='-', linewidth=0.5)
      plt.axhline(y = i + 0.5, color='gray', linestyle='-', linewidth=0.5)

    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, format(confusion[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion[i, j] > thresh else "black")

    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.show()