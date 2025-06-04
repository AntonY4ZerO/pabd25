import argparse
import os
import cianparser
import pandas as pd
import logging

moscow_parser = cianparser.CianParser(location="Москва")


def parse_cian(pages):
    "Parsing data from cian into CSVs"
    # Инициализация парсера
    moscow_parser = cianparser.CianParser(location="Москва")

    # Путь к папке для сохранения данных
    save_dir = "Zinyakov/data/raw"
    os.makedirs(save_dir, exist_ok=True)

    print("Начат сбор данных...")

    # Сбор данных по количеству комнат от 1 до 4
    for n_rooms in range(1, 5):
        print(f"Сбор данных для квартир с {n_rooms} комнатами...")

        # Путь к файлу
        file_path = os.path.join(save_dir, f"zin_{n_rooms}_rooms.csv")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pages",
        type=int,
        help="Amount of pages to parse",
    )
    args = parser.parse_args()
    parse_cian(args.pages)
