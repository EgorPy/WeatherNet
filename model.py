""" Обработка погодных данных и обучения модели нейросети """

import matplotlib.pyplot as plt
from tensorflow import keras
from config import *
import numpy as np
import sqlite3
import os


def train_model(data, epochs=500):
    """ Обучает LSTM-модель и оценивает точность """

    x = data[:-1].reshape(-1, 12, 1)  # 2014-2023 (9 лет)
    y = data[1:].reshape(-1, 12)  # 2015-2024 (9 лет)

    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=(12, 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(12)
    ])

    model.compile(optimizer="adam", loss="mae")
    history = model.fit(x, y, epochs=epochs, batch_size=8, verbose=0)

    predictions = model.predict(x)
    mae = np.mean(np.abs(y - predictions))
    mse = np.mean((y - predictions) ** 2)
    rmse = np.sqrt(mse)

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    model.save(MODEL_PATH)
    save_loss_chart(history)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2_score}


def predict_next_year(data):
    """ Предсказывает температуру следующего года используя обученную модель """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не обучена.")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mae")

    last_year = data[-1].reshape(1, 12, 1)
    prediction = model.predict(last_year)[0]

    actual_2024 = data[-1]  # = y
    mae = np.mean(np.abs(prediction - actual_2024))
    mse = np.mean((actual_2024 - prediction) ** 2)
    rmse = np.sqrt(mse)

    ss_total = np.sum((actual_2024 - np.mean(actual_2024)) ** 2)
    ss_residual = np.sum((actual_2024 - prediction) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    last_year = data[-1].reshape(1, 12, 1)
    return {
        "prediction": np.round(model.predict(last_year)[0], 2),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2_score
    }


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


def plot_temperature(predictions):
    """ Создаёт график температур """

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
    plt.plot(months, predictions, marker='o', linestyle='-', color='r', label="Predicted Temperature")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title("Predicted Monthly Temperatures for 2025")
    plt.legend()
    plt.grid(True)
    plt.savefig(PREDICTION_PLOT)
    plt.show()


def save_prediction(predictions):
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
    plt.plot(months, predictions, marker='o', linestyle='-', color='r', label="Predicted Temperature")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title("Predicted Monthly Temperatures for 2025")
    plt.legend()
    plt.grid(True)
    plt.savefig(PREDICTION_PLOT)


def plot_loss_chart(history):
    """ Создаёт график потерь """

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


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
