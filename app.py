import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# ---------------------------
# Load Dataset
# ---------------------------
dataset = load_diabetes()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

X = df
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

model = DecisionTreeRegressor(
    criterion='absolute_error',
    max_depth=5,
    max_features='sqrt',
    splitter='random'
)

model.fit(X_train, Y_train)

# ---------------------------
# HEADER
# ---------------------------
st.title("🩺 Diabetes Prediction App")
st.markdown("### Predict disease progression using ML")

st.markdown("---")

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("📥 Patient Inputs")

def user_input():
    age = st.sidebar.slider("Age", float(X.age.min()), float(X.age.max()))
    sex = st.sidebar.slider("Sex", float(X.sex.min()), float(X.sex.max()))
    bmi = st.sidebar.slider("BMI", float(X.bmi.min()), float(X.bmi.max()))
    bp = st.sidebar.slider("Blood Pressure", float(X.bp.min()), float(X.bp.max()))
    s1 = st.sidebar.slider("S1", float(X.s1.min()), float(X.s1.max()))
    s2 = st.sidebar.slider("S2", float(X.s2.min()), float(X.s2.max()))
    s3 = st.sidebar.slider("S3", float(X.s3.min()), float(X.s3.max()))
    s4 = st.sidebar.slider("S4", float(X.s4.min()), float(X.s4.max()))
    s5 = st.sidebar.slider("S5", float(X.s5.min()), float(X.s5.max()))
    s6 = st.sidebar.slider("S6", float(X.s6.min()), float(X.s6.max()))

    return np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

input_data = user_input()

# ---------------------------
# PREDICTION SECTION (CENTERED)
# ---------------------------
st.subheader("🔍 Prediction")

if st.button("Predict"):
    prediction = model.predict(input_data)

    st.success(f"Predicted Score: {prediction[0]:.2f}")
    st.info("Higher value = more severe diabetes progression")

st.markdown("---")

# ---------------------------
# INPUT SUMMARY (CLEAN TABLE)
# ---------------------------
st.subheader("📋 Input Summary")

input_df = pd.DataFrame(input_data, columns=X.columns)
st.dataframe(input_df)

st.markdown("---")

# ---------------------------
# FEATURE IMPORTANCE (SEPARATE SECTION)
# ---------------------------
st.subheader("📈 Feature Importance")

fig, ax = plt.subplots()
ax.barh(X.columns, model.feature_importances_)
st.pyplot(fig)

st.markdown("---")

# ---------------------------
# OPTIONAL DATASET
# ---------------------------
with st.expander("📂 View Dataset"):
    st.write(df.head())