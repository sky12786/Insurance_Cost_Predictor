from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("insurance.xlsx")
TARGET_COL = "PremiumPrice"

BINARY_COLS = [
    "Diabetes",
    "BloodPressureProblems",
    "AnyTransplants",
    "AnyChronicDiseases",
    "KnownAllergies",
    "HistoryOfCancerInFamily",
]

NUMERIC_COLS = [
    "Age",
    "Height",
    "Weight",
    "NumberOfMajorSurgeries",
    "BMI",
]

FEATURE_COLS = BINARY_COLS + NUMERIC_COLS


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df = df.copy()
    df["Height_m"] = df["Height"] / 100
    df["BMI"] = df["Weight"] / (df["Height_m"] ** 2)
    return df


@st.cache_resource
def train_model():
    df = load_data()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUMERIC_COLS,
            ),
            (
                "bin",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]),
                BINARY_COLS,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }

    return model, metrics


st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")
st.title("Insurance Premium Predictor")
st.write("Enter customer details to estimate the premium price.")

if not DATA_PATH.exists():
    st.error("insurance.xlsx was not found in the current folder.")
    st.stop()

model, metrics = train_model()

with st.sidebar:
    st.header("Model Performance")
    st.metric("MAE", f"{metrics['MAE']:.2f}")
    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    st.metric("R²", f"{metrics['R2']:.3f}")

st.subheader("Customer Input Form")

with st.form("premium_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
        surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0, step=1)

    with right:
        diabetes = st.selectbox("Diabetes", [0, 1], index=0)
        bp = st.selectbox("Blood Pressure Problems", [0, 1], index=0)
        transplants = st.selectbox("Any Transplants", [0, 1], index=0)
        chronic = st.selectbox("Any Chronic Diseases", [0, 1], index=0)
        allergies = st.selectbox("Known Allergies", [0, 1], index=0)
        cancer_family = st.selectbox("History Of Cancer In Family", [0, 1], index=0)

    submitted = st.form_submit_button("Predict Premium")

if submitted:
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    input_df = pd.DataFrame([{
        "Diabetes": diabetes,
        "BloodPressureProblems": bp,
        "AnyTransplants": transplants,
        "AnyChronicDiseases": chronic,
        "KnownAllergies": allergies,
        "HistoryOfCancerInFamily": cancer_family,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "NumberOfMajorSurgeries": surgeries,
        "BMI": bmi,
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Premium Price: {prediction:,.2f}")
    st.write(f"Calculated BMI: {bmi:.2f}")

    st.subheader("Input Summary")
    st.dataframe(input_df, width="stretch")