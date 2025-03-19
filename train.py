""" Тренировка и тест модели """

from model import *

data = load_last_10()
print(data)
print(data.shape)
# print(train_model(data, 800))
raw_prediction = predict_next_year(data)
print(raw_prediction)
prediction = raw_prediction["prediction"]
print(prediction.shape)
print([round(i, 2) for i in prediction.tolist()])
plot_temperature(prediction)
