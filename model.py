""" Обработка погодных данных и обучения модели нейросети """

import matplotlib.pyplot as plt
from tensorflow import keras
from config import *
import numpy as np
import sqlite3
import os

train_progress = {
    "current": 0,
    "total": 100,
    "done": True
}


class ProgressCallback(keras.callbacks.Callback):
    """ Класс для обновления прогресса обучения модели """

    def __init__(self, total_epochs, update_callback=None):
        super().__init__()
        self.total = total_epochs
        self.update_callback = update_callback

    # В ProgressCallback
    def on_epoch_end(self, epoch, logs=None):
        """ Обновляет прогресс обучения """

        train_progress["current"] = int((epoch + 1) / self.total * 100) if self.total else 0
        train_progress["total"] = 99
        train_progress["done"] = False
        if self.update_callback:
            self.update_callback(train_progress)

    def on_train_end(self, logs=None):
        """ Обновление прогресса в конце обучения """

        train_progress["current"] = 100
        train_progress["done"] = True


def train_model(data, epochs=500, window_size=3, progress_callback=None):
    """ Обучает LSTM-модель и оценивает точность """

    x_train = []
    y_train = []

    for i in range(len(data) - window_size):
        x_train.append(data[i:i + window_size])  # (window_size, 12)
        y_train.append(data[i + window_size])  # (12,)

    x_train = np.array(x_train)  # shape: (N, window_size, 12)
    y_train = np.array(y_train)  # shape: (N, 12)

    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=(window_size, 12)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(12)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="mae")
    progress = ProgressCallback(epochs, progress_callback)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=4,
        verbose=2,
        callbacks=[progress]
    )

    predictions = model.predict(x_train)
    mae = np.mean(np.abs(y_train - predictions))
    mse = np.mean((y_train - predictions) ** 2)
    rmse = np.sqrt(mse)

    ss_total = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_residual = np.sum((y_train - predictions) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    model.save(MODEL_PATH)
    save_loss_chart(history)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2_score}


def compute_accuracy(data, window_size=3):
    """ Вычисляет точность модели для данных """

    x_train = []
    y_train = []
    for i in range(len(data) - window_size):
        x_train.append(data[i:i + window_size])
        y_train.append(data[i + window_size])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = keras.models.load_model(MODEL_PATH)
    predictions = model.predict(x_train)

    mae = np.mean(np.abs(y_train - predictions))
    mse = np.mean((y_train - predictions) ** 2)
    rmse = np.sqrt(mse)

    ss_total = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_residual = np.sum((y_train - predictions) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2_score}


def predict_year(data, target_year, window_size=3, seed=None):
    """ Предсказывает температуру выбранного года 2025 - 2030 используя обученную модель """

    if not (2024 <= target_year <= 2030):
        raise ValueError("Год должен быть в диапазоне от 2025 до 2030.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не обучена.")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mae")

    if seed is not None:
        np.random.seed(seed)

    steps = target_year - 2024
    current_data = np.copy(data)  # shape: (N, 12)

    for _ in range(steps):
        window = current_data[-window_size:]  # shape: (window_size, 12)
        window_input = window.reshape(1, window_size, 12)
        predicted = model.predict(window_input, verbose=0)[0]

        noise = np.random.normal(loc=0.0, scale=0.3, size=12)  # шум (+-0.5 градуса)
        noisy_prediction = np.clip(predicted + noise, -100, 100)

        current_data = np.vstack([current_data, noisy_prediction])

    return {
        "year": target_year,
        "prediction": current_data
    }


def get_first_year_from_txt(file_path: str, delimiter: str = " ", ignore_header: bool = False):
    """ Возвращает первый год известных данных """

    with open(file_path, "r") as file:
        file.readline() if ignore_header else None
        string = file.readline()
        d = string.find(delimiter)
        year = string[d + 1:string.find(delimiter, d + 1)]
        if year.isdigit():
            return int(year)


def load_data_from_txt(file_path: str, delimiter: str = " ", ignore_header: bool = False):
    """ Загружает данные о погоде из txt файла """

    with open(file_path, "r") as file:
        file.readline() if ignore_header else None
        data = [list(map(float, row.strip().split(delimiter))) for row in file.readlines()]
    return np.array(data)[:, 2:]


def load_last_10():
    """ Загружает температурные данные за последние 10 лет из бд """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT month1, month2, month3, month4, month5, month6, "
                   "month7, month8, month9, month10, month11, month12 FROM temps ORDER BY year DESC LIMIT 10;")
    data = np.array(cursor.fetchall(), dtype=np.float32)
    conn.close()
    return data


def load_first_10():
    """ Загружает температурные данные за первые 10 лет из бд """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT month1, month2, month3, month4, month5, month6, "
                   "month7, month8, month9, month10, month11, month12 FROM temps ORDER BY year ASC LIMIT 10;")
    data = np.array(cursor.fetchall(), dtype=np.float32)
    conn.close()
    return data


def save_prediction(predictions, year):
    """ Сохраняет график температур """

    if os.path.exists(PREDICTION_PLOT):
        os.remove(PREDICTION_PLOT)
    months_names = ["Январь",
                    "Февраль",
                    "Март",
                    "Апрель",
                    "Май",
                    "Июнь",
                    "Июль",
                    "Август",
                    "Сентябрь",
                    "Октябрь",
                    "Ноябрь",
                    "Декабрь"]
    months = [months_names[i] for i in range(12)]
    plt.figure(figsize=(10, 5))
    plt.plot(months, predictions[-1], marker='o', linestyle='-', color='r', label="Predicted Temperature")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Predicted Monthly Temperatures for {year}")
    plt.legend()
    plt.grid(True)
    plt.savefig(PREDICTION_PLOT)


def save_loss_chart(history):
    """ Сохраняет график потерь """

    if os.path.exists(LOSS_PLOT):
        os.remove(LOSS_PLOT)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT)
