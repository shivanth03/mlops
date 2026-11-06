!pip install -q scikit-learn prometheus-client

import logging
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prometheus_client import start_http_server, Gauge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

# Prometheus metrics
MSE = Gauge('model_mse', 'Model Mean Squared Error')
LATENCY = Gauge('model_latency', 'Model prediction latency')

# Load dataset
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
logging.info("Starting model training...")
model = LinearRegression()
model.fit(X_train, y_train)
logging.info("Model training completed.")

# Make predictions and calculate latency
start = time.time()
y_pred = model.predict(X_test)
latency = time.time() - start

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

logging.info("Mean Squared Error: %s", mse)
logging.info("Prediction latency: %.6f seconds", latency)

# Check model performance
if mse <= 3500:
    logging.info("MSE within acceptable range (threshold=3500.0).")
else:
    logging.warning("MSE exceeds acceptable threshold (threshold=3500.0).")

# Update Prometheus metrics
MSE.set(mse)
LATENCY.set(latency)

# Start Prometheus server
start_http_server(8000)

print("Metrics exposed at http://127.0.0.1:8000/metrics")

import requests

try:
    response = requests.get("http://127.0.0.1:8000/metrics")
    print("Local server status:", response.status_code)
    print(response.text[:1500])
except Exception as e:
    print("Server is not running:", e)

from google.colab import output

proxy_url = output.eval_js(
    "google.colab.kernel.proxyPort(8000)"
)

print("Open this URL:")
print(proxy_url.rstrip("/") + "/metrics")
