""" Модуль с логикой для предсказания температуры годов 2025 - 2030 """

from model import *


def predict_year(data, target_year, window_size=3, seed=None):
    """ Предсказывает температуру выбранного года 2025 - 2030 используя обученную модель """

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
