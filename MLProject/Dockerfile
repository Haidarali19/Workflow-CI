FROM python:3.12.7-slim

WORKDIR /app

# Install dependencies langsung agar lebih ringan
RUN pip install pandas mlflow==2.19.0 dagshub scikit-learn matplotlib

COPY . .

# Jalankan skrip modelling saat container dimulai
CMD ["python", "modelling.py"]