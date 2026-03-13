import streamlit as st
import pandas as pd
import joblib

# charger modèle et colonnes
model = joblib.load("model.pkl")
model_columns = joblib.load("columns.pkl")

st.title("AI Real Estate Prospect Predictor")

# formulaire utilisateur
age = st.number_input("Age", 18, 100)
income = st.number_input("Income")
children = st.number_input("Children", 0, 10)
budget = st.number_input("Budget")

marital_status = st.selectbox(
    "Marital Status",
    ["single","married","divorced"]
)

city = st.selectbox(
    "City",
    ["Rabat","Casablanca","Tangier","Agadir","Marrakech"]
)

property_type = st.selectbox(
    "Property Type",
    ["apartment","villa","duplex","studio"]
)

# bouton prédiction
if st.button("Predict"):

    # données utilisateur
    data = {
        "age":[age],
        "income":[income],
        "marital_status":[marital_status],
        "children":[children],
        "budget":[budget],
        "city":[city],
        "property_type":[property_type]
    }

    df = pd.DataFrame(data)

    # transformation ML
    df_ml = pd.get_dummies(df)
    df_ml = df_ml.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_ml)

    if prediction[0] == 1:
        st.success("Client likely to BUY")
    else:
        st.error("Client unlikely to buy")

    # ---------------------------
    # ajout automatique dataset
    # ---------------------------

    dataset = pd.read_csv("real_estate_prospects.csv", sep=";", encoding="utf-8-sig")

    dataset.columns = dataset.columns.str.strip()

    new_id = dataset["id"].max() + 1

    df["id"] = new_id
    df["bought"] = prediction[0]

    df = df[[
        "id",
        "age",
        "income",
        "marital_status",
        "children",
        "budget",
        "city",
        "property_type",
        "bought"
    ]]

    dataset = pd.concat([dataset, df], ignore_index=True)

    try:
        dataset.to_csv("real_estate_prospects.csv", index=False)
        st.success("Prospect ajouté au dataset")
    except PermissionError:
        st.error("Fermez le fichier CSV dans Excel puis réessayez")

# ---------------------------
# afficher dataset
# ---------------------------

st.subheader("Prospects Dataset")

data = pd.read_csv("real_estate_prospects.csv", sep=";", encoding="utf-8-sig")

data.columns = data.columns.str.strip()

st.dataframe(data)