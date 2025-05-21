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


def train_model(model_prefix):
    raw_dir = "../pabd25/data/raw"
    model_dir = "../pabd25/models"
    os.makedirs(model_dir, exist_ok=True)

    for room_count in range(1, 5):
        file_path = os.path.join(raw_dir, f"{room_count}_rooms.csv")
        if not os.path.exists(file_path):
            logging.warning(f"Файл не найден: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            df = df[["total_meters", "floor", "floors_count", "rooms_count", "price"]]
            df = df.dropna()
            df = df[(df["total_meters"] <= 300) & (df["price"] < 100_000_000)]
        except Exception as e:
            logging.error(f"Ошибка обработки {file_path}: {e}")
            continue

        if df.empty:
            logging.warning(f"Нет подходящих данных для {room_count} комнат")
            continue

        df = df.sort_index()
        X = df[["total_meters", "floor", "floors_count", "rooms_count"]]
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))

        logging.info(
            f"[{room_count} комн] MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}"
        )

        # Автоопределение версии
        base_name = f"{model_prefix}_{room_count}room"
        existing = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pkl"))
        versions = [
            int(f.split("_v")[-1].replace(".pkl", "")) for f in existing if "_v" in f
        ]
        next_version = max(versions, default=0) + 1

        model_path = os.path.join(model_dir, f"{base_name}_v{next_version}.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Модель сохранена: {model_path}")


def test_model(model_name):
    model_dir = "../pabd25/models"

    # Тестовые входные данные по количеству комнат
    test_data_by_rooms = {
        1: [{"total_meters": 30, "floor": 1, "floors_count": 3, "rooms_count": 1}],
        2: [{"total_meters": 45, "floor": 2, "floors_count": 5, "rooms_count": 2}],
        3: [{"total_meters": 60, "floor": 4, "floors_count": 9, "rooms_count": 3}],
        4: [{"total_meters": 80, "floor": 6, "floors_count": 12, "rooms_count": 4}],
    }

    for room_count in range(1, 5):
        pattern = os.path.join(model_dir, f"{model_name}_{room_count}room_v*.pkl")
        matching_files = glob.glob(pattern)

        if not matching_files:
            logging.warning(f"Нет модели для {room_count} комнат по шаблону: {pattern}")
            continue

        # Выбираем файл с максимальной версией
        latest_model_file = max(
            matching_files, key=lambda f: int(f.split("_v")[-1].replace(".pkl", ""))
        )

        logging.info(f"Загрузка модели: {latest_model_file}")
        model = joblib.load(latest_model_file)

        input_data = test_data_by_rooms[room_count]
        input_df = pd.DataFrame(input_data)

        predicted_prices = model.predict(input_df)

        logging.info(f"=== Предсказания для {room_count}-комнатных квартир ===")
        for features, price in zip(input_data, predicted_prices):
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
    parse_cian(args.pages)
    preprocess_data()
    train_model(args.model)
    test_model(args.model)
