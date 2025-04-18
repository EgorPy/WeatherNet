# Прогнозирование средней температуры  

Этот проект использует нейросеть для предсказания средней температуры по месяцам для 2025 года.

## Возможности  
- Загрузка данных из `.txt`  
- Обучение модели и сохранение в `.keras`  
- Построение графиков температур и потерь  
- Предсказание температуры на 2025-2030 года  
- Оценка точности (MAE, MSE, RMSE, R²)  
- Веб-интерфейс на FastAPI

## Запуск проекта
- Зайдите в релизы и скачайте build.zip
- Распакуйте архив build.zip
- Запустите файл main.exe

## Запуск кода
```sh
pip install -r requirements.txt
cd WeatherNet/
python main.py
```

## Источник
Для обучения используются реальные данные с 2014 по 2024 года с [meteo.ru](https://meteo.ru).  
