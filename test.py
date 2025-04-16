import requests

# URL вашего эндпоинта
url = "http://127.0.0.1:8000/train/"  # Замените на актуальный URL вашего сервера

# Данные для отправки в POST-запросе
data = {
    "use_existing": "false",  # Например, использовать обучение новой модели
    "epochs": 500  # Количество эпох
}

# Отправка POST-запроса
response = requests.post(url, data=data)

# Проверка ответа сервера
if response.status_code == 200:
    print("Запрос прошел успешно.")
    print("Ответ от сервера:")
    print(response.json())  # Печать ответа сервера в формате JSON
else:
    print(f"Ошибка при запросе. Статус код: {response.status_code}")
    print("Ответ от сервера:")
    print(response.text)
