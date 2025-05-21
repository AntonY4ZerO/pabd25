import os
import logging
import glob
import joblib
import pandas as pd
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
    test_model(args.model)
