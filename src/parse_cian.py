import datetime
import os

import cianparser
import pandas as pd

moscow_parser = cianparser.CianParser(location="Москва")


def main():
    """
    Парсинг квартир в Москве по количеству комнат (1-4) и сохранение в отдельные CSV-файлы
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs("data/raw", exist_ok=True)

    for n_rooms in range(1, 5):
        print(f"Сбор данных для квартир с {n_rooms} комнатами...")

        csv_path = f"data/raw/{n_rooms}_rooms_{t}.csv"
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 5,
                "object_type": "secondary",
            },
        )

        df = pd.DataFrame(data)
        df.to_csv(csv_path, encoding="utf-8", index=False)

        print(f"Файл сохранён: {csv_path}")


if __name__ == "__main__":
    main()
