"""
🐦 Projet Data Science : Analyse de Sentiments sur des Tweets
Dataset : Twitter US Airline Sentiment / Sentiment140 (Kaggle)
"""

# ==================== IMPORT DES BIBLIOTHÈQUES ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# NLP & Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# NLTK pour le traitement du texte
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("✅ Bibliothèques importées avec succès")
print("✅ Ressources NLTK téléchargées\n")

# ==================== 1. CHARGEMENT ET EXPLORATION ====================
print("=" * 70)
print("📊 ÉTAPE 1 : CHARGEMENT ET EXPLORATION DES DONNÉES")
print("=" * 70)

# Charger le dataset
# Pour Twitter US Airline Sentiment : colonnes = ['airline_sentiment', 'text', ...]
# Adapter selon ton dataset
try:
    df = pd.read_csv('tweets.csv', encoding='utf-8')
except:
    try:
        df = pd.read_csv('tweets.csv', encoding='latin-1')
    except:
        print("⚠️  Fichier non trouvé. Création d'un dataset de démonstration...")
        # Dataset de démonstration
        df = pd.DataFrame({
            'text': [
                'I love this airline! Great service',
                'Terrible experience, never flying again',
                'Flight was okay, nothing special',
                'Amazing staff and comfortable seats',
                'Delayed flight, very disappointed',
                'Not bad but could be better',
                'Excellent customer service!',
                'Worst airline ever, lost my baggage',
                'Average flight experience',
                'Highly recommend this airline!'
            ] * 100,
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                         'neutral', 'positive', 'negative', 'neutral', 'positive'] * 100
        })

print(f"\n📌 Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n📋 Aperçu des premières lignes :")
print(df.head(10))

print(f"\n🔍 Informations sur les colonnes :")
print(df.info())

# Identifier la colonne de sentiment (peut varier selon le dataset)
sentiment_col = None
text_col = None

for col in df.columns:
    if 'sentiment' in col.lower():
        sentiment_col = col
    if 'text' in col.lower() or 'tweet' in col.lower():
        text_col = col

if sentiment_col is None:
    sentiment_col = df.columns[0]
if text_col is None:
    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

print(f"\n✅ Colonne sentiment détectée : '{sentiment_col}'")
print(f"✅ Colonne texte détectée : '{text_col}'")

# Renommer pour uniformiser
df = df.rename(columns={sentiment_col: 'sentiment', text_col: 'text'})

# Vérifier les valeurs manquantes
print(f"\n❌ Valeurs manquantes :")
print(df.isnull().sum())

# Supprimer les lignes avec valeurs manquantes
df = df.dropna(subset=['text', 'sentiment'])

# Supprimer les doublons
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\n🔄 Doublons supprimés : {duplicates}")

# Répartition des sentiments
print(f"\n📊 Répartition des sentiments :")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
print(f"\n{sentiment_counts / len(df) * 100}")

# Longueur des tweets
df['text_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

print(f"\n📏 Statistiques sur la longueur des tweets :")
print(df[['text_length', 'word_count']].describe())

# ==================== 2. NETTOYAGE DU TEXTE ====================
print("\n" + "=" * 70)
print("🧹 ÉTAPE 2 : NETTOYAGE ET PRÉTRAITEMENT DU TEXTE")
print("=" * 70)

def clean_text(text):
    """
    Nettoie un tweet en :
    - Convertissant en minuscules
    - Supprimant les URLs
    - Supprimant les mentions (@user)
    - Supprimant les hashtags (#)
    - Supprimant la ponctuation et caractères spéciaux
    - Supprimant les espaces multiples
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("🔧 Application du nettoyage...")
df['clean_text'] = df['text'].apply(clean_text)

# Afficher des exemples avant/après
print("\n📝 Exemples de nettoyage :")
for i in range(3):
    print(f"\nAvant : {df['text'].iloc[i][:80]}...")
    print(f"Après : {df['clean_text'].iloc[i][:80]}...")

# ==================== 3. PRÉTRAITEMENT LINGUISTIQUE (NLP) ====================
print("\n" + "=" * 70)
print("🔤 ÉTAPE 3 : PRÉTRAITEMENT LINGUISTIQUE")
print("=" * 70)

# Suppression des stopwords
stop_words = set(stopwords.words('english'))
# Ajouter des mots spécifiques si nécessaire (ne pas être trop agressif)
# custom_stopwords = {'flight', 'airline'}  
# stop_words.update(custom_stopwords)

def remove_stopwords(text):
    """Supprime les stopwords"""
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words and len(word) > 2])

print("🔧 Suppression des stopwords...")
df['processed_text'] = df['clean_text'].apply(remove_stopwords)

# Lemmatization (optionnel, peut ralentir le traitement)
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    """Lemmatize le texte"""
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

print("🔧 Lemmatization en cours...")
df['final_text'] = df['processed_text'].apply(lemmatize_text)

# Filtrer les tweets vides après nettoyage
df = df[df['final_text'].str.len() > 0]

print(f"\n✅ Nombre de tweets après nettoyage : {len(df)}")
print(f"✅ Exemple de texte final : '{df['final_text'].iloc[0]}'")

# ==================== 4. VISUALISATION EXPLORATOIRE ====================
print("\n" + "=" * 70)
print("📊 ÉTAPE 4 : VISUALISATION EXPLORATOIRE")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))

# 1. Répartition des sentiments
ax1 = plt.subplot(2, 3, 1)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(sentiment_counts)]
sentiment_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('📊 Répartition des Sentiments', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Nombre de tweets')
plt.xticks(rotation=45)
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Distribution de la longueur des tweets
ax2 = plt.subplot(2, 3, 2)
plt.hist(df['word_count'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {df["word_count"].mean():.1f}')
plt.title('📏 Distribution de la Longueur des Tweets', fontsize=14, fontweight='bold')
plt.xlabel('Nombre de mots')
plt.ylabel('Fréquence')
plt.legend()

# 3. Longueur moyenne par sentiment
ax3 = plt.subplot(2, 3, 3)
avg_length = df.groupby('sentiment')['word_count'].mean().sort_values()
avg_length.plot(kind='barh', color='coral', edgecolor='black')
plt.title('📊 Longueur Moyenne par Sentiment', fontsize=14, fontweight='bold')
plt.xlabel('Nombre moyen de mots')
plt.ylabel('Sentiment')

# 4. Top 15 mots les plus fréquents (global)
ax4 = plt.subplot(2, 3, 4)
all_words = ' '.join(df['final_text']).split()
word_freq = Counter(all_words).most_common(15)
words, counts = zip(*word_freq)
plt.barh(range(len(words)), counts, color='lightgreen', edgecolor='black')
plt.yticks(range(len(words)), words)
plt.xlabel('Fréquence')
plt.title('🔤 Top 15 Mots les Plus Fréquents', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 5. Distribution des sentiments (pie chart)
ax5 = plt.subplot(2, 3, 5)
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('📊 Proportion des Sentiments', fontsize=14, fontweight='bold')
plt.ylabel('')

# 6. Nombre de caractères par sentiment
ax6 = plt.subplot(2, 3, 6)
df.boxplot(column='text_length', by='sentiment', ax=ax6, patch_artist=True)
plt.title('📏 Distribution de la Longueur (caractères) par Sentiment', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.xlabel('Sentiment')
plt.ylabel('Nombre de caractères')

plt.tight_layout()
plt.savefig('twitter_eda.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques sauvegardés : twitter_eda.png")
plt.show()

# ==================== WORDCLOUDS ====================
print("\n📊 Génération des WordClouds...")

fig = plt.figure(figsize=(18, 6))

sentiments = df['sentiment'].unique()
for idx, sent in enumerate(sentiments[:3], 1):
    ax = plt.subplot(1, 3, idx)
    text_data = ' '.join(df[df['sentiment'] == sent]['final_text'])
    
    if len(text_data) > 0:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='Set2',
            max_words=100
        ).generate(text_data)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'☁️  WordCloud - {sent.upper()}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('twitter_wordclouds.png', dpi=300, bbox_inches='tight')
print("✅ WordClouds sauvegardés : twitter_wordclouds.png")
plt.show()

# ==================== 5. TRANSFORMATION EN VECTEURS ====================
print("\n" + "=" * 70)
print("🔢 ÉTAPE 5 : TRANSFORMATION EN VECTEURS (TF-IDF)")
print("=" * 70)

# TF-IDF Vectorization avec paramètres adaptés
# Ajuster max_features en fonction de la taille du dataset
max_features = min(5000, len(df) * 10)

vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(1, 2),  # Unigrammes et bigrammes
    min_df=1,  # Au moins 1 document (adapté pour petits datasets)
    max_df=0.9  # Ignorer les mots trop fréquents
)

print("🔧 Transformation TF-IDF en cours...")
X = vectorizer.fit_transform(df['final_text'])
y = df['sentiment']

print(f"\n✅ Matrice TF-IDF créée : {X.shape}")
print(f"✅ Nombre de features : {X.shape[1]}")
print(f"✅ Nombre d'échantillons : {X.shape[0]}")

# Afficher les mots les plus importants
feature_names = vectorizer.get_feature_names_out()
print(f"\n📝 Exemples de features : {list(feature_names[:10])}")

# ==================== 6. MODÉLISATION ====================
print("\n" + "=" * 70)
print("🤖 ÉTAPE 6 : MODÉLISATION MACHINE LEARNING")
print("=" * 70)

# Division train/test avec gestion des petits datasets
# Vérifier si le dataset est assez grand pour stratifier
min_samples = y.value_counts().min()
test_size = 0.2

if len(y) < 30 or min_samples < 3:
    print(f"\n⚠️  Dataset trop petit ({len(y)} échantillons). Ajustement des paramètres...")
    test_size = 0.3 if len(y) >= 10 else 0.2
    stratify_param = None  # Pas de stratification pour très petits datasets
else:
    stratify_param = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify_param
)

print(f"\n✅ Taille du set d'entraînement : {X_train.shape}")
print(f"✅ Taille du set de test : {X_test.shape}")

# ========== Modèle 1 : Naive Bayes ==========
print("\n📊 Modèle 1 : Multinomial Naive Bayes")
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("\n📊 Résultats Naive Bayes :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"F1-Score (weighted) : {f1_score(y_test, y_pred_nb, average='weighted'):.4f}")
print("\n📋 Classification Report :")
print(classification_report(y_test, y_pred_nb))

# ========== Modèle 2 : Logistic Regression ==========
print("\n📈 Modèle 2 : Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n📊 Résultats Logistic Regression :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score (weighted) : {f1_score(y_test, y_pred_lr, average='weighted'):.4f}")
print("\n📋 Classification Report :")
print(classification_report(y_test, y_pred_lr))

# ========== Modèle 3 : Linear SVM ==========
print("\n🎯 Modèle 3 : Linear SVM")
svm_model = LinearSVC(random_state=42, max_iter=1000)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\n📊 Résultats Linear SVM :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score (weighted) : {f1_score(y_test, y_pred_svm, average='weighted'):.4f}")
print("\n📋 Classification Report :")
print(classification_report(y_test, y_pred_svm))

# ==================== 7. VISUALISATION DES RÉSULTATS ====================
print("\n" + "=" * 70)
print("📊 ÉTAPE 7 : VISUALISATION DES RÉSULTATS")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))

# 1. Matrice de confusion - Naive Bayes
ax1 = plt.subplot(2, 3, 1)
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Naive Bayes', fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

# 2. Matrice de confusion - Logistic Regression
ax2 = plt.subplot(2, 3, 2)
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

# 3. Matrice de confusion - SVM
ax3 = plt.subplot(2, 3, 3)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Linear SVM', fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

# 4. Comparaison des modèles - Accuracy
ax4 = plt.subplot(2, 3, 4)
models = ['Naive Bayes', 'Logistic Reg', 'Linear SVM']
accuracies = [
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_svm)
]
colors_bar = ['#3498db', '#2ecc71', '#9b59b6']
bars = plt.bar(models, accuracies, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.title('📊 Comparaison Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
             f'{acc:.3f}', ha='center', fontweight='bold', fontsize=11)

# 5. Comparaison F1-Score
ax5 = plt.subplot(2, 3, 5)
f1_scores = [
    f1_score(y_test, y_pred_nb, average='weighted'),
    f1_score(y_test, y_pred_lr, average='weighted'),
    f1_score(y_test, y_pred_svm, average='weighted')
]
bars = plt.bar(models, f1_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.title('📊 Comparaison F1-Score', fontsize=14, fontweight='bold')
plt.ylabel('F1-Score (weighted)')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
    plt.text(bar.get_x() + bar.get_width()/2, f1 + 0.02,
             f'{f1:.3f}', ha='center', fontweight='bold', fontsize=11)

# 6. Top mots importants (Logistic Regression)
ax6 = plt.subplot(2, 3, 6)
if hasattr(lr_model, 'coef_'):
    # Pour un problème binaire ou premier sentiment
    coef_index = 0 if len(lr_model.coef_) > 0 else 0
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(lr_model.coef_[coef_index])
    }).sort_values('importance', ascending=False).head(15)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'], color='coral')
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance (valeur absolue)')
    plt.title('Top 15 Mots Importants (LR)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('twitter_model_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques sauvegardés : twitter_model_results.png")
plt.show()

# ==================== 8. TEST DE PRÉDICTION ====================
print("\n" + "=" * 70)
print("🧪 ÉTAPE 8 : TEST DE PRÉDICTION SUR NOUVEAUX TWEETS")
print("=" * 70)

def predict_sentiment(tweet, model=lr_model, vectorizer=vectorizer):
    """Prédit le sentiment d'un tweet"""
    cleaned = clean_text(tweet)
    processed = remove_stopwords(cleaned)
    lemmatized = lemmatize_text(processed)
    vectorized = vectorizer.transform([lemmatized])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0] if hasattr(model, 'predict_proba') else None
    return prediction, proba

# Exemples de tweets
test_tweets = [
    "This airline is amazing! Best flight ever 😊",
    "Terrible service, my flight was delayed for 5 hours",
    "The flight was okay, nothing special",
    "I love flying with this company, great experience!",
    "Worst airline ever, lost my luggage"
]

print("\n🐦 Prédictions sur de nouveaux tweets :\n")
for tweet in test_tweets:
    pred, proba = predict_sentiment(tweet)
    print(f"Tweet: {tweet[:60]}...")
    print(f"→ Sentiment prédit: {pred}")
    if proba is not None:
        print(f"→ Confiance: {max(proba)*100:.1f}%")
    print()

# ==================== 9. CONCLUSION ====================
print("=" * 70)
print("🎯 CONCLUSION ET RÉSUMÉ DU PROJET")
print("=" * 70)

print(f"""
📊 RÉSUMÉ DES RÉSULTATS :

1️⃣ Dataset :
   - {len(df)} tweets analysés après nettoyage
   - {len(df['sentiment'].unique())} classes de sentiments : {', '.join(df['sentiment'].unique())}
   - Longueur moyenne : {df['word_count'].mean():.1f} mots par tweet

2️⃣ Prétraitement :
   - Nettoyage : URLs, mentions, hashtags, ponctuation
   - Suppression des stopwords
   - Lemmatization
   - Vectorisation TF-IDF : {X.shape[1]} features

3️⃣ Performances des modèles :
   
   📊 Naive Bayes :
   - Accuracy : {accuracy_score(y_test, y_pred_nb):.4f}
   - F1-Score : {f1_score(y_test, y_pred_nb, average='weighted'):.4f}
   
   📈 Logistic Regression :
   - Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}
   - F1-Score : {f1_score(y_test, y_pred_lr, average='weighted'):.4f}
   
   🎯 Linear SVM :
   - Accuracy : {accuracy_score(y_test, y_pred_svm):.4f}
   - F1-Score : {f1_score(y_test, y_pred_svm, average='weighted'):.4f}

4️⃣ Meilleur modèle :
   - {models[np.argmax(accuracies)]} avec {max(accuracies):.2%} d'accuracy

💡 PISTES D'AMÉLIORATION :

   ✓ Utiliser des modèles pré-entraînés (BERT, RoBERTa, DistilBERT)
   ✓ Créer une interface web avec Streamlit/Gradio
   ✓ Enrichir le dataset via l'API Twitter/X
   ✓ Ajouter l'analyse d'emojis et leur sentiment
   ✓ Implémenter un système de détection de sarcasme
   ✓ Analyser les sentiments par sujet (aspect-based sentiment)
   ✓ Créer un dashboard temps réel
   ✓ Ajouter du deep learning (LSTM, GRU, Transformer)

📝 COMPÉTENCES DÉMONTRÉES :

   • Natural Language Processing (NLP)
   • Text preprocessing & cleaning
   • Feature extraction (TF-IDF)
   • Machine Learning (Classification)
   • Data Visualization
   • Model evaluation & comparison
   • Python (pandas, scikit-learn, nltk)

🎉 PROJET TERMINÉ AVEC SUCCÈS !
""")

print("=" * 70)
print("\n💾 Fichiers générés :")
print("   - twitter_eda.png : Analyse exploratoire")
print("   - twitter_wordclouds.png : Nuages de mots")
print("   - twitter_model_results.png : Résultats des modèles")
print("\n✨ Tu peux maintenant ajouter ce projet à ton CV/portfolio !")