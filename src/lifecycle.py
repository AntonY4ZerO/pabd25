"This is full life cycle for ai model"

import os
import cianparser
import pandas as pd
import glob
import numpy as np
import os
import joblib
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import argparse


# Настройка логгера
os.makedirs("../pabd25/notebooks/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../pabd25/notebooks/logs/lifecycle.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def parse_cian(pages):
    "Parsing data from cian into CSVs"
    # Инициализация парсера
    moscow_parser = cianparser.CianParser(location="Москва")

    # Путь к папке для сохранения данных
    save_dir = "data/raw"
    os.makedirs(save_dir, exist_ok=True)

    print("Начат сбор данных...")

    # Сбор данных по количеству комнат от 1 до 4
    for n_rooms in range(1, 5):
        print(f"Сбор данных для квартир с {n_rooms} комнатами...")

        # Путь к файлу
        file_path = os.path.join(save_dir, f"{n_rooms}_rooms.csv")

        # Получение данных
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": pages,
                "object_type": "secondary",
            },
        )

        # Преобразование в DataFrame
        df = pd.DataFrame(data)

        # Если файл существует, добавляем новые данные
        if os.path.exists(file_path):
            # Читаем существующие данные
            existing_df = pd.read_csv(file_path)
            # Добавляем новые данные
            df = pd.concat([existing_df, df], ignore_index=True)

        # Сохраняем (или перезаписываем файл)
        df.to_csv(file_path, encoding="utf-8", index=False)
        print(f"Данные для {n_rooms} комнат сохранены в {file_path}")


def preprocess_data():
    "Preprocessing collected data from parse_cian()"
    # Папка с исходными файлами
    raw_data_path = "../pabd25/data/raw"
    file_list = glob.glob(os.path.join(raw_data_path, "*.csv"))

    # Чтение и объединение всех файлов
    dataframes = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Ошибка при чтении {file}: {e}")

    # Объединение всех данных
    main_dataframe = pd.concat(dataframes, ignore_index=True)

    # Уникальный ID и отбор нужных колонок
    main_dataframe["url_id"] = main_dataframe["url"].map(
        lambda x: x.split("/")[-2] if isinstance(x, str) else None
    )
    df = (
        main_dataframe[
            ["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]
        ]
        .dropna()
        .set_index("url_id")
    )

    # Фильтрация по площади и цене
    df = df[(df["total_meters"] <= 300) & (df["price"] < 100_000_000)]

    # Сохранение объединённого и очищенного файла
    os.makedirs("../pabd25/data/processed", exist_ok=True)
    df.to_csv("../pabd25/data/processed/merged_cleaned.csv", encoding="utf-8")


def train_model(model_name):
    # Загрузка данных
    data = pd.read_csv("../pabd25/data/processed/merged_cleaned.csv")

    # Удаление пропусков (если есть)
    data = data.dropna(
        subset=["total_meters", "floor", "floors_count", "rooms_count", "price"]
    )

    # Сортировка данных по индексу (или можно по дате, если есть такой столбец)
    data = data.sort_index(ascending=True)

    # Разделение данных на признаки и цель
    X = data[["total_meters", "floor", "floors_count", "rooms_count"]]
    y = data["price"]

    # Разделение на обучающую и тестовую выборки (по 80% и 20% соответственно)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)

    # Метрики
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    # Логирование метрик
    logging.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logging.info(f"Корень из MSE (RMSE): {rmse:.2f}")
    logging.info(f"Коэффициент детерминации R²: {r2:.6f}")
    logging.info(f"Средняя абсолютная ошибка (MAE): {mae:.2f} рублей")
    logging.info(f"Коэффициент при площади: {model.coef_[0]:.2f}")
    logging.info(f"Свободный член (intercept): {model.intercept_:.2f}")

    # Сохранение модели
    os.makedirs("../pabd25/models", exist_ok=True)
    model_path = f"../pabd25/models/{model_name}.pkl"

    joblib.dump(model, model_path)

    logging.info(f"Модель сохранена: {model_path}")


def test_model(model_name):
    # Загрузка модели
    model_path = f"../pabd25/models/{model_name}.pkl"
    model = joblib.load(model_path)
    logging.info(f"Модель загружена из: {model_path}")

    # Массив входных данных
    data = [
        {"total_meters": 45, "floor": 2, "floors_count": 5, "rooms_count": 2},
        {"total_meters": 60, "floor": 4, "floors_count": 9, "rooms_count": 3},
        {"total_meters": 30, "floor": 1, "floors_count": 3, "rooms_count": 1},
        {"total_meters": 80, "floor": 6, "floors_count": 12, "rooms_count": 4},
    ]

    input_df = pd.DataFrame(data)

    # Предсказание
    predicted_prices = model.predict(input_df)

    # Вывод и логирование
    logging.info("=== Предсказания модели по массиву данных ===")
    for features, price in zip(data, predicted_prices):
        log_msg = (
            f"Площадь: {features['total_meters']} м², "
            f"Комнат: {features['rooms_count']}, "
            f"Этаж: {features['floor']}/{features['floors_count']} → "
            f"Цена: {price:,.0f} ₽"
        )
        logging.info(log_msg)


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument(
        "-p",
        "--pages",
        type=int,
        help="Amount of pages to parse",
    )
    args = parser.parse_args()
    # parse_cian(args.pages)
    preprocess_data()
    train_model(args.model)
    test_model(args.model)
