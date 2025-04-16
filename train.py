""" Тренировка и тест модели """

from main import load_session
from model import *

session = load_session()
file_path = session["file_path"]
data = load_data_from_txt(file_path, session["delimiter"], session["ignore_header"])


# # train model
# print(data)
# print(data.shape)
# print(train_model(data, 500))

# # predict years 2025 - 2030
def py(year):
    """ Predict year """

    raw_prediction = predict_year(data, year)
    prediction = raw_prediction["prediction"]
    print(prediction[-1])


for i in range(2024, 2031):
    py(i)

# 2025 [-5.45, -4.16, 0.58, 7.97, 12.79, 17.42, 18.32, 17.66, 13.36, 7.41, 0.51, -4.35]
# 2026 [-5.33, -4.02, 0.87, 7.28, 12.8, 17.26, 18.2, 17.37, 13.13, 7.28, 0.34, -4.45]
# 2027 [-5.59, -4.32, 0.85, 7.39, 12.91, 16.99, 18.81, 17.46, 13.2, 7.45, 0.39, -4.2]
# 2028 [-5.45, -4.45, 0.95, 7.61, 12.56, 17.33, 18.73, 17.16, 12.97, 7.45, 0.84, -4.46]
# 2029 [-5.93, -4.01, 0.87, 7.09, 12.86, 17.12, 18.82, 17.49, 13.43, 7.4, 0.77, -4.18]
# 2030 [-5.2, -4.12, 0.46, 7.61, 12.67, 16.83, 19.17, 17.42, 12.8, 7.84, 0.59, -4.29]
