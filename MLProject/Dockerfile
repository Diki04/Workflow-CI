# 1. Mulai dari base image Python
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy file environment
COPY conda.yaml .

# 4. Install conda dan environment
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda env create -f conda.yaml

# 5. Aktifkan environment
SHELL ["conda", "run", "-n", "mlproject-gym-env", "/bin/bash", "-c"]

# 6. Copy model yang sudah dilatih (dari 'mlruns')
COPY mlruns /app/mlruns

# 7. Tentukan port
EXPOSE 5000

# 8. Perintah untuk menjalankan model server
CMD ["mlflow", "models", "serve", "-m", "/app/mlruns/0/[RUN_ID]/artifacts/model", "-h", "0.0.0.0", "-p", "5000"]