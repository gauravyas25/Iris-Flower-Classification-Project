# 🌸 Iris Flower Classification Project

🔗 **Live Streamlit App:**  
https://iris-flower-classification-project-100.streamlit.app/

---

## 📌 Project Overview

This project demonstrates a **complete end-to-end Machine Learning classification workflow**
using the **Iris Flower Dataset**.  
The main objective is to **train, evaluate, compare, and deploy multiple supervised classification models**
and analyze their performance using **visualizations**.

The project covers all major stages of a Machine Learning pipeline:
- Data loading and preprocessing
- Model training
- Model evaluation
- Performance visualization
- Web-based deployment using Streamlit

---

## 🎯 Problem Statement

Given the physical measurements of an iris flower:
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

the task is to **predict the species of the flower**, which belongs to one of the following classes:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

This is a **multi-class supervised classification problem**.

---

## 📊 Dataset Description

- **Dataset Name:** Iris Dataset  
- **Total Samples:** 150  
- **Number of Features:** 4 (numerical)  
- **Target Variable:** Species (categorical)  
- **Number of Classes:** 3  

The Iris dataset is a standard dataset in Machine Learning because:
- It is clean and well-structured
- It contains balanced classes
- It is ideal for understanding classification algorithms

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. **Data Loading**
   - The dataset was loaded using Pandas.
   - A preview of the dataset is displayed in the Streamlit app.

2. **Label Encoding**
   - The target variable (`Species`) is categorical.
   - It was converted into numerical form using **Label Encoding** to make it compatible with ML models.

3. **Train-Test Split**
   - The dataset was split into **50% training data and 50% testing data**.
   - Training data is used to train the models.
   - Testing data is used to evaluate model performance on unseen data.

---

## 🤖 Machine Learning Models Implemented

The following supervised learning algorithms were implemented and compared:

### 1️⃣ Logistic Regression
- A linear classification algorithm.
- Works well when classes are linearly separable.
- Used as a strong baseline model.

### 2️⃣ K-Nearest Neighbors (KNN)
- A distance-based, non-parametric algorithm.
- Classifies data points based on the majority class of nearest neighbors.
- Effective in capturing local patterns in data.

### 3️⃣ Naive Bayes (Gaussian)
- A probabilistic classifier based on Bayes’ theorem.
- Assumes independence among features.
- Performs well on small and clean datasets.

All models were trained on the same training dataset to ensure a **fair comparison**.

---

## 📈 Model Evaluation Metrics

### ✅ Accuracy Score
Accuracy measures the **overall correctness** of the model and is calculated as:

(Number of correct predictions) / (Total predictions)

Accuracy is suitable for this project because:
- The dataset is balanced
- All classes are equally important

---

## 📊 Accuracy Comparison Visualization

**What this visualization represents:**
- Each bar represents a Machine Learning model.
- The height of the bar corresponds to the model’s accuracy on test data.

**Why it is useful:**
- Enables quick comparison between models
- Helps identify the best-performing algorithm
- Makes performance differences easy to interpret visually

---

## 📉 Confusion Matrix Visualization

A **confusion matrix** provides a detailed view of model predictions.

### Structure:
- Rows → Actual class labels
- Columns → Predicted class labels

### Interpretation:
- Diagonal values represent correct predictions
- Off-diagonal values represent misclassifications

### Importance:
- Accuracy alone does not show class-wise errors
- Confusion matrices help analyze which classes are confused
- Essential for understanding detailed model behavior

Separate confusion matrices are generated for **each model**.

---

## 📊 Visual Analysis Summary

- Accuracy charts provide a **high-level performance comparison**
- Confusion matrices provide **class-wise performance insights**
- Together, these visualizations ensure **transparent and interpretable evaluation**

---

## 🛠 Tools & Technologies Used

- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  
- **Web Deployment:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 🚀 Deployment Details

- The Machine Learning logic was converted from a Jupyter Notebook into a **Streamlit web application**
- The application is deployed using **Streamlit Cloud**
- Users can interactively view results, metrics, and visualizations through a web browser

🔗 **Live App:**  
https://iris-flower-classification-project-100.streamlit.app/

---

## 🧠 Learning Outcomes

Through this project, the following concepts were practiced:

- End-to-end Machine Learning workflow
- Supervised classification algorithms
- Model comparison and evaluation
- Performance visualization techniques
- Deployment of ML models using Streamlit

This project serves as a **strong foundational Machine Learning project** and can be extended further with:
- User input-based predictions
- Cross-validation analysis
- Additional evaluation metrics

---

## 👨‍💻 Author

**Gaurav Vyas**  
Machine Learning & Full Stack Developer  

---

⭐ If you found this project useful, feel free to star the repository!
