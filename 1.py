!git config --global user.name "MLOPS Student"
!git config --global user.email "aditya@example.com"
!git init -q
!echo "Introduction to mlops" > README.md
!git add README.md
!git commit -m "Initial Commit"
!git log --oneline

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(n_estimators=3,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred)*100,"%")
