FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN pip install --no-cache-dir "setuptools<60" "wheel" "cython<3" "numpy==1.26.4"

RUN pip install --no-cache-dir --no-build-isolation lightfm==1.17


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]