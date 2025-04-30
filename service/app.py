from flask import Flask, jsonify, render_template, request
import logging
import joblib
import pandas as pd

app = Flask(__name__)
# === Загрузка модели ===
model = joblib.load('models/linear_regression_model.pkl') #notebooks\models\linear_regression_model.pkl

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Формат логирования
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Лог в файл
file_handler = logging.FileHandler('service/app_logs.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Лог в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    params = request.get_json()


    try:
        area = int(params['area'])
        rooms = int(params['rooms'])
        totalfloors = int(params['totalfloors'])
        floor = int(params['floor'])

        # Валидация
        if not (1 <= rooms <= 4):
            error_msg = 'Количество комнат должно быть от 1 до 4 включительно.'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400

        if totalfloors > 50:
            error_msg = 'Этажей в доме не может быть больше 50.'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400

        if floor > totalfloors:
            error_msg = 'Этаж квартиры не может быть больше, чем этажей в доме.'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400

        # Подготовка данных и предсказание
        input_df = pd.DataFrame({'total_meters': [area]})
        predicted_price = model.predict(input_df)[0]

        logger.info(f'Полученные данные: Площадь = {area}, Комнаты = {rooms}, Этажей в доме = {totalfloors}, Этаж квартиры = {floor}')
        logger.info(f'Расчётная цена: {predicted_price}')
        logger.info('Статус: success')
        return {'Цена': predicted_price}

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f'Ошибка обработки данных: {e}')
        return jsonify({'error': 'Некорректные входные данные.'}), 400

if __name__ == '__main__':
    app.run(debug=True)