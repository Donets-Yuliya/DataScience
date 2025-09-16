from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train(X, y):

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def predict(model, X):

    return model.predict(X)

def evaluate(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Визуализация истинных и предсказанных значений
def plot_predictions(y_true, y_pred, num_points=50):
    plt.figure(figsize=(18, 10))
    plt.scatter(range(num_points), y_true[:num_points], color='blue', label='Истинные значения')
    plt.scatter(range(num_points), y_pred[:num_points], color='red', label='Предсказанные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Значение charges')
    plt.title(f'Истинные и предсказанные значения charges (первые {num_points} точек)')
    plt.legend()
    plt.show()