"""Download multiple model files from S3 storage"""

import os
import argparse
from dotenv import dotenv_values
import boto3

BUCKET_NAME = "pabd25"
YOUR_SURNAME = "zinyakov"

# Названия моделей, которые нужно загрузить
MODEL_FILES = [
    "xgboost_model_1room_v2.pkl",
    "xgboost_model_2room_v2.pkl",
    "xgboost_model_3room_v2.pkl",
    "xgboost_model_4room_v2.pkl",
]

# Убедимся, что папка models существует
os.makedirs("models", exist_ok=True)

# Загрузка конфигурации из .env
config = dotenv_values(".env")


def download_models():
    # Создаём клиента S3 с доступами
    client = boto3.client(
        "s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
    )

    # Загружаем каждый файл
    for model_file in MODEL_FILES:
        object_name = f"{YOUR_SURNAME}/models/{model_file}"
        local_path = f"models/{model_file}"
        try:
            print(f"Загружается {model_file} из {object_name} ...")
            client.download_file(BUCKET_NAME, object_name, local_path)
            print(f"Успешно: {model_file}")
        except Exception as e:
            print(f"Ошибка при загрузке {model_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from S3")
    args = parser.parse_args()
    download_models()
