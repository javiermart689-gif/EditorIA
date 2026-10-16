# Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# Instalar FFmpeg a nivel del sistema operativo (Esto soluciona el error)
RUN apt-get update && apt-get install -y ffmpeg

# Crear el directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto
COPY . .

# Comando para iniciar FastAPI en Railway
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
