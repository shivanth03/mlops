import os
import subprocess


with open("train.py", "w") as f:
    f.write("""
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes()
X, y = data.data, data.target

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "diabetes_model.joblib")
print("Model trained and saved successfully!")
""")


with open("test_model.py", "w") as f:
    f.write("""
import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("diabetes_model.joblib")
    prediction = model.predict(np.zeros((1, 10)))
    assert isinstance(prediction[0], (float, np.floating))

def test_model_outputshape():
    model = joblib.load("diabetes_model.joblib")
    prediction = model.predict(np.zeros((5, 10)))
    assert prediction.shape == (5,)
""")


print("Files:", os.listdir())


subprocess.run(["python", "train.py"], check=True)


subprocess.run(["python", "-m", "pytest", "test_model.py", "-v"])

!pytest test_model.py -v
