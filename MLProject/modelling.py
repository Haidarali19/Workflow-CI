import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Inisialisasi DagsHub
dagshub.init(repo_owner='Haidarali19', repo_name='Eksperimen_SML_M.-Haidar', mlflow=True)

# Mengaktifkan Autolog
mlflow.sklearn.autolog()

# Load Data
df = pd.read_csv('house_prices_preprocessing.csv')
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Menghitung Prediksi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mlflow.log_metric("manual_mae", mae)

print(f"Baseline Model Trained. MAE: {mae}")