FROM python:3.10.6-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000


CMD ["python", "-m", "project.app.fast"]
