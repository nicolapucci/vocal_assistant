FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .env .

COPY clients/ clients/
COPY managers/ managers/
COPY tasks/ tasks/

COPY decorators.py .

COPY app.py .

ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["flask","run","--host=0.0.0.0","--port=5000"]