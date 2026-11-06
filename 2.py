
!pip install -q scikit-learn joblib flask


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify


data = load_diabetes()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, pred))


joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')


loaded_pred = model.predict(X_test)
print("Loaded Model MSE:", mean_squared_error(y_test, loaded_pred))


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    result = model.predict([features])
    return jsonify({'prediction': float(result[0])})

client = app.test_client()
response = client.post('/predict', json={'features': X_test[0].tolist()})

print(response.json)
