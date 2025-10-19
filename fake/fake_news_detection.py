import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Chargement et exploration
df = pd.read_csv('train.csv')
print(df.head())
print(df.isnull().sum())

# Fusionner titre + texte
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

# 2️⃣ Nettoyage du texte
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)          # liens
    text = re.sub(r'[^a-z\s]', '', text)         # caractères spéciaux
    return text

df['content'] = df['content'].apply(clean_text)

# 3️⃣ Prétraitement NLP
nltk.download('stopwords')
stop = set(stopwords.words('english'))
df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# 4️⃣ Séparation des données
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# 5️⃣ Transformation en vecteurs TF-IDF
tfidf = TfidfVectorizer(max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6️⃣ Entraînement du modèle
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# 7️⃣ Évaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8️⃣ Visualisation
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Purples')
plt.title("Matrice de confusion - Détection de Fake News")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()
