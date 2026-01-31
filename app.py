import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("iris.csv")
df.head()

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])


X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

models = {'Logistic Regression': lr, 'KNN': knn, 'Naive Bayes': nb}

for name, model in models.items():
    y_pred = model.predict(X)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y, y_pred))


accuracies = {}

models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "Naive Bayes": nb
}

# Evaluate on TEST data
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# Bar plot for accuracy comparison
plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of ML Models on Iris Dataset")
plt.ylim(0, 1)
plt.show()


models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "Naive Bayes": nb
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred
    )
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.show()