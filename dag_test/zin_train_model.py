import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import glob
import joblib
import argparse


def train_model(model_prefix):
    raw_dir = "Zinyakov/data/raw"
    model_dir = "Zinyakov/models"
    os.makedirs(model_dir, exist_ok=True)

    for room_count in range(1, 5):
        file_path = os.path.join(raw_dir, f"zin_{room_count}_rooms.csv")
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            df = df[["total_meters", "floor", "floors_count", "rooms_count", "price"]]
            df = df.dropna()
            df = df[(df["total_meters"] <= 300) & (df["price"] < 100_000_000)]
        except Exception as e:
            print(f"Ошибка обработки {file_path}: {e}")
            continue

        if df.empty:
            print(f"Нет подходящих данных для {room_count} комнат")
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

        print(
            f"[{room_count} комн] MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}"
        )

        base_name = f"zin_{model_prefix}_{room_count}room"
        existing = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pkl"))
        versions = [
            int(f.split("_v")[-1].replace(".pkl", "")) for f in existing if "_v" in f
        ]
        next_version = max(versions, default=0) + 1

        model_path = os.path.join(model_dir, f"{base_name}_v{next_version}.pkl")
        joblib.dump(model, model_path)
        print(f"Модель сохранена: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name")
    args = parser.parse_args()
    train_model(args.model)
