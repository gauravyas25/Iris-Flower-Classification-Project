import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Iris ML App", layout="centered")
st.title("🌸 Iris Flower Classification App")
st.write("App started successfully ✅")

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Iris.csv")

df = pd.read_csv(csv_path)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- PREPROCESSING ----------------
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# ---------------- TRAIN MODELS ----------------
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)

models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "Naive Bayes": nb
}

# ---------------- ACCURACY OUTPUT ----------------
st.subheader("Model Accuracy Comparison")

accuracies = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    st.write(f"**{name} Accuracy:** {acc}")

# ---------------- ACCURACY BAR CHART ----------------
fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values())
ax.set_ylim(0, 1)
ax.set_xlabel("Models")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison of ML Models")

st.pyplot(fig)

# ---------------- CONFUSION MATRICES ----------------
st.subheader("Confusion Matrices")

for name, model in models.items():
    y_pred = model.predict(X_test)

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax_cm
    )
    ax_cm.set_title(f"{name} Confusion Matrix")

    st.pyplot(fig_cm)

st.success("App executed completely 🎉")
