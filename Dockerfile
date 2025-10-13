FROM python:3.12-slim

WORKDIR /app

# Создание директории для логов
RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs  # Права на запись для всех пользователей

RUN pip cache purge

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY ./models ./models
COPY ./src ./src
COPY ./app ./app
COPY ./train_data ./train_data

CMD ["python", "./app/app.py"]
