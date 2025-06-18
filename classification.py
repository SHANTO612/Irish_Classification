import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("🌸 Iris Species Predictor")
st.markdown("This app uses a **Random Forest Classifier** to predict the species of an iris flower based on its measurements.")

# Load data with caching
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train the model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar for input
st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), step=0.1
)
sepal_width = st.sidebar.slider(
    "Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), step=0.1
)
petal_length = st.sidebar.slider(
    "Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), step=0.1
)
petal_width = st.sidebar.slider(
    "Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), step=0.1
)

# Predict
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]
prediction_proba = model.predict_proba(input_data)

# Display result
st.subheader("🌼 Prediction Result")
st.success(f"Predicted Iris Species: **{predicted_species}**")

st.subheader("📊 Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.bar_chart(proba_df.T)

# Optional: show raw data
if st.checkbox("Show training data"):
    st.subheader("🔍 Iris Dataset (first 10 rows)")
    st.dataframe(df.head(10))
