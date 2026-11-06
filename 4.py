%pip install -q mlflow==2.22.4


import mlflow 
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data=load_diabetes()
x=pd.DataFrame(data.data, columns=data.feature_names)
y=data.target

x_train,x_test, y_train, y_test= train_test_split(x,y,test_size=.2,random_state=42)

mlflow.set_experiment("diabetes-regression")

models={
    "LinearRegression":LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
  with mlflow.start_run(run_name=name):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)
    mlflow.log_params(model.get_params())
    mlflow.log_metric("MSE",mse)

    mlflow.sklearn.log_model(model, "model")

    print(name, "MSE:", round(mse, 2))

print("Experiment tracking completed!")

