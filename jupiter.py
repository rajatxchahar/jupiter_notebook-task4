# 📍 Spam Email Detection using Scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 1. Load Dataset
# Dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ['label', 'message']

# 🔹 2. Preprocess
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔹 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 🔹 4. Vectorize Text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔹 5. Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 🔹 6. Predictions
y_pred = model.predict(X_test_vec)

# 🔹 7. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
