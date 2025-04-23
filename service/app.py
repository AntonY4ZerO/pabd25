from flask import Flask, json, jsonify, render_template, request
import logging

app = Flask(__name__)

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            return jsonify({'error': 'Количество комнат должно быть от 1 до 4 включительно.'}), 400
        if totalfloors > 50:
            return jsonify({'error': 'Этажей в доме не может быть больше 50.'}), 400
        if floor > totalfloors:
            return jsonify({'error': 'Этаж квартиры не может быть больше, чем этажей в доме.'}), 400

        price = area * rooms * totalfloors * floor

        logger.info(f'Полученные данные: Площадь = {area}, Комнаты = {rooms}, Этажей в доме = {totalfloors}, Этаж квартиры = {floor}')
        logger.info(f'Расчётная цена: {price}')

        return {'Цена': price}

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f'Ошибка обработки данных: {e}')
        return jsonify({'error': 'Некорректные входные данные.'}), 400

if __name__ == '__main__':
    app.run(debug=True)