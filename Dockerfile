# 1. Imagen base oficial de Python ligera
FROM python:3.11-slim

# 2. Evitar que Python escriba archivos .pyc y habilitar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiar e instalar dependencias primero (aprovecha el caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el código de la aplicación, scripts y los modelos entrenados
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# 6. Crear carpeta para logs de monitoreo
RUN mkdir -p monitoring

# 7. Exponer el puerto de FastAPI
EXPOSE 8000

# 8. Comando para iniciar el servidor en producción
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]