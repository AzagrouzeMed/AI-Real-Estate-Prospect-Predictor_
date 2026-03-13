import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv("real_estate_prospects.csv")

data = data.drop("id", axis=1)

X = data.drop("bought", axis=1)
y = data["bought"]

X = pd.get_dummies(X)

# sauvegarder les colonnes
joblib.dump(X.columns, "columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model entraîné avec succès")