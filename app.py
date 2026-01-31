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


st.markdown("""
## 📌 Project Overview

This project presents a **complete end-to-end Machine Learning classification workflow**
implemented using the **Iris Flower Dataset**.  
The goal of the project is to **train, evaluate, compare, and deploy multiple classification models**
while understanding their behavior through **visual analysis**.

The project is designed to demonstrate **core ML concepts**, including data preprocessing,
model training, performance evaluation, and result interpretation.

---

## 🎯 Problem Statement

Given the physical measurements of an iris flower:
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

the task is to **predict the species of the flower**, which can be:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

This is a **multi-class supervised classification problem**.

---

## 📊 Dataset Description

- Dataset: **Iris Dataset**
- Total Samples: 150
- Number of Features: 4 (numerical)
- Target Classes: 3 flower species

The Iris dataset is widely used in Machine Learning because:
- It is clean and well-structured
- Classes are reasonably separable
- It is ideal for understanding classification algorithms

---

## ⚙️ Data Preprocessing Steps

1. **Data Loading**
   - The dataset is loaded using Pandas.
   - A preview of the dataset is displayed to understand its structure.

2. **Label Encoding**
   - The target variable (Species) is categorical.
   - It is converted into numerical form using **Label Encoding** so that ML models can process it.

3. **Train-Test Split**
   - The dataset is split into **50% training and 50% testing data**.
   - Training data is used to learn patterns.
   - Testing data is used to evaluate performance on unseen data.

---

## 🤖 Machine Learning Models Implemented

### 1️⃣ Logistic Regression
- A linear classification algorithm.
- Works well when classes are linearly separable.
- Serves as a strong baseline model.

### 2️⃣ K-Nearest Neighbors (KNN)
- A distance-based, non-parametric algorithm.
- Classifies a sample based on the majority class of its nearest neighbors.
- Captures local data patterns effectively.

### 3️⃣ Naive Bayes (Gaussian)
- A probabilistic classifier based on Bayes’ theorem.
- Assumes feature independence.
- Performs well on small and clean datasets.

All models are trained on the same training data for a **fair comparison**.

---

## 📈 Model Evaluation Metrics

### ✅ Accuracy Score
Accuracy measures the **overall correctness** of the model.

It is calculated as:
> (Number of correct predictions) / (Total predictions)

Accuracy is suitable here because:
- The dataset is balanced
- All classes are equally important

---

## 📊 Accuracy Comparison Bar Chart (Visualization Meaning)

**What this graph shows:**
- Each bar represents a Machine Learning model.
- The height of the bar represents the model’s accuracy on test data.

**Why it is important:**
- Allows quick comparison between models.
- Helps identify which model performs best.
- Makes performance differences visually intuitive.

---

## 📉 Confusion Matrix (Visualization Meaning)

A **confusion matrix** provides a detailed breakdown of model predictions.

### Structure:
- Rows → Actual class labels
- Columns → Predicted class labels

### What it tells us:
- Correct predictions appear on the **diagonal**
- Misclassifications appear off the diagonal
- Helps identify which classes are being confused

### Why it is useful:
- Accuracy alone does not show class-wise errors
- Confusion matrices help evaluate **model reliability**
- Essential for understanding model behavior in detail

Each model has its **own confusion matrix** for individual analysis.

---

## 📊 Visual Analysis Summary

- The accuracy chart provides a **high-level comparison**
- Confusion matrices provide **fine-grained insights**
- Combined, these visualizations ensure **transparent and interpretable evaluation**

---

## 🛠 Tools & Technologies Used

- Programming Language: **Python**
- Data Handling: Pandas, NumPy
- Machine Learning: Scikit-learn
- Visualization: Matplotlib
- Deployment: Streamlit
- Version Control: Git & GitHub

---

## 🚀 Deployment Details

- The Machine Learning pipeline was converted from a Jupyter Notebook into a **Streamlit web application**
- The application is deployed using **Streamlit Cloud**
- This allows users to view results and visualizations interactively through a web browser

---

## 🧠 Learning Outcomes

Through this project, the following concepts were practiced:
- End-to-end ML workflow
- Supervised classification
- Model comparison
- Performance visualization
- Real-world ML deployment

This project serves as a **foundation for more advanced Machine Learning applications**.
""")


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
