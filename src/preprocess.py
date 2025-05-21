import glob
import pandas as pd
import os
import logging

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


if __name__ == "__main__":
    preprocess_data()
