from flask import Flask, jsonify, render_template, request
import logging
import joblib
import pandas as pd
import subprocess
import argparse

app = Flask(__name__)

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Формат логирования
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Лог в файл
file_handler = logging.FileHandler("service/app_logs.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Лог в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def format_price(price):
    millions = int(price // 1_000_000)
    thousands = int((price % 1_000_000) // 1_000)
    return (
        f"{millions} миллион {thousands} тысяч"
        if millions > 0
        else f"{thousands} тысяч"
    )


# Загрузка всех моделей при старте
def load_models():
    models = {}
    for i in range(1, 5):
        model_name = f"xgboost_model_{i}room_v1"
        try:
            models[i] = joblib.load(f"models/{model_name}.pkl")
            logger.info(f"Модель {model_name} успешно загружена.")
        except FileNotFoundError:
            logger.warning(f"Модель {model_name} не найдена.")
    return models


# Инициализация моделей
models = load_models()


# Аргументы командной строки
def get_model_name_from_args():
    parser = argparse.ArgumentParser(
        description="Запуск приложения для предсказания цены"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="Название модели (без расширения .pkl)"
    )
    args = parser.parse_args()
    return args.model


# Маршрут для отображения формы
@app.route("/")
def index():
    return render_template("index.html")


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
def process_numbers():
    params = request.get_json()

    try:
        area = int(params["area"])
        rooms = int(params["rooms"])
        totalfloors = int(params["totalfloors"])
        floor = int(params["floor"])

        # Валидация
        if not (1 <= rooms <= 4):
            error_msg = "Количество комнат должно быть от 1 до 4 включительно."
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400

        if totalfloors > 50:
            error_msg = "Этажей в доме не может быть больше 50."
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400

        if floor > totalfloors:
            error_msg = "Этаж квартиры не может быть больше, чем этажей в доме."
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400

        # Проверка наличия соответствующей модели
        if rooms not in models:
            error_msg = f"Модель для {rooms} комнат не найдена."
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500

        # Подготовка данных
        input_df = pd.DataFrame(
            [
                {
                    "total_meters": area,
                    "floor": floor,
                    "floors_count": totalfloors,
                    "rooms_count": rooms,
                }
            ]
        )

        # Предсказание с соответствующей моделью
        predicted_price = models[rooms].predict(input_df)[0]
        formatted_price = format_price(predicted_price)

        logger.info(f"Используется модель: xgboost_model_{rooms}room_v1")
        logger.info(
            f"Полученные данные: Площадь = {area}, Комнаты = {rooms}, Этажей в доме = {totalfloors}, Этаж квартиры = {floor}"
        )
        logger.info(f"Расчётная цена: {predicted_price}")
        logger.info("Статус: success")
        return {"Цена": formatted_price}

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Ошибка обработки данных: {e}")
        return jsonify({"error": "Некорректные входные данные."}), 400


if __name__ == "__main__":
    app.run(debug=True)
