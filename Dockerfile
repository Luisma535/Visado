# Usar una imagen base de Python
FROM python:3.11-slim

# Instalar dependencias del sistema incluyendo Chrome (versión corregida)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    && mkdir -p /etc/apt/keyrings \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub > /etc/apt/keyrings/google-chrome-key.asc \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome-key.asc] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Crear directorio para estado local
RUN mkdir -p estado_local

# Comando para ejecutar la aplicación
CMD ["python", "bot_visado.py"]
