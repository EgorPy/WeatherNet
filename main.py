""" Главный исполняемый файл """

from model import load_data_from_txt, train_model, predict_year, save_prediction, train_progress, compute_accuracy
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from config import UPLOAD_DIR, SESSION_FILE, MODEL_PATH
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import BackgroundTasks
import webbrowser
import threading
import uvicorn
import json
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="pages")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_session(file_path, delimiter, ignore_header):
    """ Сохраняет небольшие пользовательские данные """

    with open(SESSION_FILE, "w") as f:
        json.dump({
            "file_path": file_path,
            "delimiter": delimiter,
            "ignore_header": ignore_header
        }, f)


def load_session():
    """ Загружает пользовательские данные """

    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ Главная страница """

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """ Страница загрузки данных """

    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/train/", response_class=HTMLResponse)
async def train_page(request: Request):
    """ Страница обучения """

    return templates.TemplateResponse("train.html", {"request": request})


@app.get("/predict/", response_class=HTMLResponse)
async def predict_page(request: Request):
    """ Страница предсказания """

    return templates.TemplateResponse("predict.html", {"request": request, "years": list(range(2024, 2031))})


@app.get("/plots/", response_class=HTMLResponse)
async def plots(request: Request):
    """ Страница графиков """

    return templates.TemplateResponse("plots.html", {"request": request})


@app.get("/model/", response_class=HTMLResponse)
async def viewmodel(request: Request):
    """ Страница просмотра и скачивания текущей модели """

    return templates.TemplateResponse("model.html", {"request": request})


@app.get("/api/model/", response_class=JSONResponse)
async def list_models():
    """ API endpoint для получения информации о текущей модели """

    if not os.path.exists(MODEL_PATH):
        return JSONResponse(content=[])
    files = os.listdir(os.path.dirname(MODEL_PATH))

    return JSONResponse({
        "files": files
    })


@app.get("/model/{filename}")
async def get_upload(filename: str):
    """ API endpoint для скачивания указанной модели """

    filepath = os.path.join("models", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return JSONResponse({"error": "File not found"}, status_code=404)


@app.get("/uploads/", response_class=HTMLResponse)
async def uploads(request: Request):
    """ Страница просмотра загруженных данных """

    return templates.TemplateResponse("uploads.html", {"request": request})


@app.get("/api/uploads/", response_class=JSONResponse)
async def list_uploads():
    """ API endpoint для получения списка загруженных файлов """

    if not os.path.exists(UPLOAD_DIR):
        return JSONResponse(content=[])
    files = os.listdir(UPLOAD_DIR)
    return JSONResponse(content=files)


@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """ API endpoint для скачивания указанного файла """

    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)


@app.post("/upload/")
async def upload_file(
        file: UploadFile = File(...),
        delimiter: str = Query(","),
        ignore_header: str = Form("false")
):
    """ API endpoint для загрузки файла """

    ignore_header_parsed = ignore_header.lower() == "true"
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    content = await file.read()
    if ignore_header_parsed:
        content = b"\n".join(content.split(b"\n")[1:])
    content = content.replace(b",", delimiter.encode())

    with open(file_path, "wb") as savefile:
        savefile.write(content)

    save_session(file_path, delimiter, ignore_header_parsed)

    try:
        data = load_data_from_txt(file_path, delimiter, ignore_header_parsed)
        return JSONResponse({"message": "Файл загружен", "data_shape": data.shape})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/train/progress/")
def get_train_progress():
    """ API endpoint для получения текущего прогресса обучения модели """

    return JSONResponse(content={
        "done": train_progress.get("done", False),
        "current": train_progress.get("current", 0),
        "accuracy": train_progress.get("accuracy", None),
        "total": train_progress.get("total", 0)
    })


@app.get("/accuracy/")
async def get_accuracy():
    """ API endpoint для оценки точности модели """

    session_data = load_session()
    if "file_path" not in session_data:
        return JSONResponse(status_code=400, content={"detail": "Нет данных для обучения"})

    file_path = session_data["file_path"]
    data = load_data_from_txt(file_path, session_data["delimiter"], session_data["ignore_header"])

    accuracy = compute_accuracy(data)

    return JSONResponse({
        "message": "Модель обучена и сохранена",
        "mae": accuracy["mae"],
        "mse": accuracy["mse"],
        "rmse": accuracy["rmse"],
        "r2": accuracy["r2"]
    })


@app.post("/train/")
async def train(background_tasks: BackgroundTasks, use_existing: bool = Query(True), epochs: int = Form(500)):
    """ API endpoint для обучения модели """

    session_data = load_session()
    if "file_path" not in session_data:
        return JSONResponse(status_code=400, content={"detail": "Нет данных для обучения"})

    file_path = session_data["file_path"]
    data = load_data_from_txt(file_path, session_data["delimiter"], session_data["ignore_header"])

    def train_task():
        """ Обучение модели """

        accuracy = train_model(data, epochs)
        train_progress["done"] = True
        train_progress.update(accuracy)
        print(f"Обучение завершено. Точность: {accuracy}")

    train_progress.clear()
    train_progress["done"] = False
    background_tasks.add_task(train_task)

    if use_existing:
        return JSONResponse({"message": "Загружена ранее обученная модель"})
    return JSONResponse({"message": "Обучение начато", "accuracy": None})


@app.post("/predict/")
async def predict(year: int = Form(...)):
    """ API endpoint для предсказания температуры следующего года """

    session_data = load_session()
    if "file_path" not in session_data:
        raise HTTPException(status_code=400, detail="Нет данных для предсказания")

    file_path = session_data["file_path"]
    data = load_data_from_txt(file_path, session_data["delimiter"], session_data["ignore_header"])
    raw_prediction = predict_year(data, year)
    prediction = raw_prediction["prediction"]
    save_prediction(prediction, year)

    return JSONResponse({
        "prediction": [round(i, 2) for i in prediction[-1].tolist()],
        "mae": None,
        "mse": None,
        "rmse": None,
        "r2": None
    })


def start_server():
    """ Запускает сервер """

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


def run():
    """ Обрабатывает запуск сервера """

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    webbrowser.open("http://127.0.0.1:8000")
    server_thread.join()


if __name__ == '__main__':
    run()
