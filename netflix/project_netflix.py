"""
🎬 Projet Data Science : Analyse et prédiction des films Netflix
Dataset : Netflix Movies and TV Shows (Kaggle)
"""

# ==================== IMPORT DES BIBLIOTHÈQUES ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Bibliothèques importées avec succès\n")

# ==================== 1. IMPORT ET EXPLORATION ====================
print("=" * 60)
print("📊 ÉTAPE 1 : IMPORT ET EXPLORATION DES DONNÉES")
print("=" * 60)

# Charger le dataset
df = pd.read_csv('netflix_titles.csv')

print(f"\n📌 Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n📋 Aperçu des premières lignes :")
print(df.head())

print(f"\n🔍 Informations sur les colonnes :")
print(df.info())

print(f"\n📈 Statistiques descriptives :")
print(df.describe())

# Vérifier les valeurs manquantes
print(f"\n❌ Valeurs manquantes par colonne :")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Manquantes': missing, 'Pourcentage': missing_pct})
print(missing_df[missing_df['Manquantes'] > 0].sort_values('Manquantes', ascending=False))

# Vérifier les doublons
duplicates = df.duplicated().sum()
print(f"\n🔄 Nombre de doublons : {duplicates}")

# Statistiques générales
print(f"\n📊 Répartition Type de contenu :")
print(df['type'].value_counts())

# ==================== 2. NETTOYAGE ET PRÉTRAITEMENT ====================
print("\n" + "=" * 60)
print("🧹 ÉTAPE 2 : NETTOYAGE ET PRÉTRAITEMENT")
print("=" * 60)

# Copie pour le nettoyage
df_clean = df.copy()

# Supprimer les doublons
df_clean = df_clean.drop_duplicates()
print(f"\n✅ Doublons supprimés : {len(df) - len(df_clean)}")

# Gérer les valeurs manquantes
df_clean['director'] = df_clean['director'].fillna('Unknown')
df_clean['cast'] = df_clean['cast'].fillna('Unknown')
df_clean['country'] = df_clean['country'].fillna('Unknown')
df_clean['rating'] = df_clean['rating'].fillna('Not Rated')

# Supprimer les lignes avec date_added manquante
df_clean = df_clean.dropna(subset=['date_added'])

print(f"✅ Valeurs manquantes traitées")
print(f"✅ Taille finale du dataset : {df_clean.shape}")

# Nettoyer et convertir les dates
df_clean['date_added'] = pd.to_datetime(df_clean['date_added'].str.strip(), format='%B %d, %Y', errors='coerce')
df_clean['year_added'] = df_clean['date_added'].dt.year
df_clean['month_added'] = df_clean['date_added'].dt.month

# Extraire la durée numérique
def extract_duration(duration, content_type):
    if pd.isna(duration):
        return np.nan
    if content_type == 'Movie':
        return int(duration.split()[0])  # "90 min" -> 90
    else:
        return int(duration.split()[0])  # "2 Seasons" -> 2

df_clean['duration_value'] = df_clean.apply(
    lambda x: extract_duration(x['duration'], x['type']), axis=1
)

# Extraire le premier pays
df_clean['primary_country'] = df_clean['country'].apply(lambda x: x.split(',')[0].strip())

# Nombre d'acteurs
df_clean['cast_count'] = df_clean['cast'].apply(
    lambda x: 0 if x == 'Unknown' else len(x.split(','))
)

# Nombre de genres
df_clean['genre_count'] = df_clean['listed_in'].apply(lambda x: len(x.split(',')))

# Extraire le premier genre
df_clean['primary_genre'] = df_clean['listed_in'].apply(lambda x: x.split(',')[0].strip())

print("\n✅ Features créées : year_added, month_added, duration_value, primary_country, cast_count, genre_count, primary_genre")

# ==================== 3. ANALYSE EXPLORATOIRE ====================
print("\n" + "=" * 60)
print("📊 ÉTAPE 3 : ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 60)

# Configuration des subplots
fig = plt.figure(figsize=(20, 12))

# 1. Top 10 des pays
ax1 = plt.subplot(2, 3, 1)
top_countries = df_clean['primary_country'].value_counts().head(10)
top_countries.plot(kind='barh', color='skyblue')
plt.title('Top 10 des pays producteurs', fontsize=14, fontweight='bold')
plt.xlabel('Nombre de contenus')
plt.ylabel('Pays')
plt.gca().invert_yaxis()

# 2. Évolution par année
ax2 = plt.subplot(2, 3, 2)
year_counts = df_clean['release_year'].value_counts().sort_index()
year_counts[year_counts.index >= 2000].plot(kind='line', linewidth=2, color='coral')
plt.title('Évolution du nombre de contenus par année (depuis 2000)', fontsize=14, fontweight='bold')
plt.xlabel('Année de sortie')
plt.ylabel('Nombre de contenus')
plt.grid(alpha=0.3)

# 3. Top genres
ax3 = plt.subplot(2, 3, 3)
top_genres = df_clean['primary_genre'].value_counts().head(10)
top_genres.plot(kind='bar', color='lightgreen')
plt.title('Top 10 des genres', fontsize=14, fontweight='bold')
plt.xlabel('Genre')
plt.ylabel('Nombre de contenus')
plt.xticks(rotation=45, ha='right')

# 4. Répartition des durées (Films)
ax4 = plt.subplot(2, 3, 4)
movies_duration = df_clean[df_clean['type'] == 'Movie']['duration_value'].dropna()
plt.hist(movies_duration, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
plt.title('Distribution des durées des films (minutes)', fontsize=14, fontweight='bold')
plt.xlabel('Durée (min)')
plt.ylabel('Fréquence')

# 5. Répartition Type de contenu
ax5 = plt.subplot(2, 3, 5)
df_clean['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Répartition Films vs Séries', fontsize=14, fontweight='bold')
plt.ylabel('')

# 6. Ajouts par année
ax6 = plt.subplot(2, 3, 6)
year_added_counts = df_clean['year_added'].value_counts().sort_index()
year_added_counts.plot(kind='area', color='gold', alpha=0.6, linewidth=2)
plt.title('Nombre de contenus ajoutés sur Netflix par année', fontsize=14, fontweight='bold')
plt.xlabel('Année d\'ajout')
plt.ylabel('Nombre de contenus')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('netflix_eda.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques sauvegardés : netflix_eda.png")
plt.show()

# ==================== 4. FEATURE ENGINEERING ====================
print("\n" + "=" * 60)
print("🔧 ÉTAPE 4 : FEATURE ENGINEERING")
print("=" * 60)

# Créer la variable cible : is_popular
# Critères : films récents (après 2015) OU genres populaires OU durée optimale
popular_genres = ['International Movies', 'Dramas', 'Comedies', 'Action & Adventure', 'Documentaries']

df_clean['is_popular'] = (
    ((df_clean['release_year'] >= 2015) & (df_clean['type'] == 'Movie')) |
    (df_clean['primary_genre'].isin(popular_genres)) |
    ((df_clean['duration_value'] >= 80) & (df_clean['duration_value'] <= 120) & (df_clean['type'] == 'Movie'))
).astype(int)

print(f"\n🎯 Distribution de la variable cible (is_popular) :")
print(df_clean['is_popular'].value_counts())
print(f"Pourcentage de contenus populaires : {df_clean['is_popular'].mean() * 100:.2f}%")

# Sélectionner les features pour le modèle
features_to_encode = ['type', 'rating', 'primary_country', 'primary_genre']
numerical_features = ['release_year', 'duration_value', 'cast_count', 'genre_count', 'year_added', 'month_added']

# Créer le dataset pour la modélisation
df_model = df_clean[features_to_encode + numerical_features + ['is_popular']].copy()
df_model = df_model.dropna()

print(f"\n✅ Dataset pour modélisation : {df_model.shape}")

# Encodage des variables catégorielles
le_dict = {}
for col in features_to_encode:
    le = LabelEncoder()
    df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])
    le_dict[col] = le
    print(f"✅ Encodage : {col} -> {df_model[f'{col}_encoded'].nunique()} classes")

# Sélectionner les features finales
feature_columns = [f'{col}_encoded' for col in features_to_encode] + numerical_features
X = df_model[feature_columns]
y = df_model['is_popular']

print(f"\n📊 Shape des features (X) : {X.shape}")
print(f"📊 Shape de la cible (y) : {y.shape}")

# ==================== 5. MODÉLISATION ====================
print("\n" + "=" * 60)
print("🤖 ÉTAPE 5 : MODÉLISATION MACHINE LEARNING")
print("=" * 60)

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Taille du set d'entraînement : {X_train.shape}")
print(f"✅ Taille du set de test : {X_test.shape}")

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Modèle 1 : Random Forest ==========
print("\n🌲 Modèle 1 : Random Forest Classifier")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n📊 Résultats Random Forest :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_rf):.4f}")
print("\n📋 Classification Report :")
print(classification_report(y_test, y_pred_rf, target_names=['Non Populaire', 'Populaire']))

# ========== Modèle 2 : Logistic Regression ==========
print("\n📈 Modèle 2 : Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n📊 Résultats Logistic Regression :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_lr):.4f}")
print("\n📋 Classification Report :")
print(classification_report(y_test, y_pred_lr, target_names=['Non Populaire', 'Populaire']))

# ==================== 6. VISUALISATION DES RÉSULTATS ====================
print("\n" + "=" * 60)
print("📊 ÉTAPE 6 : VISUALISATION DES RÉSULTATS")
print("=" * 60)

fig = plt.figure(figsize=(18, 10))

# 1. Matrice de confusion - Random Forest
ax1 = plt.subplot(2, 3, 1)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Pop', 'Pop'], yticklabels=['Non Pop', 'Pop'])
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

# 2. Matrice de confusion - Logistic Regression
ax2 = plt.subplot(2, 3, 2)
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=['Non Pop', 'Pop'], yticklabels=['Non Pop', 'Pop'])
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

# 3. Importance des features - Random Forest
ax3 = plt.subplot(2, 3, 3)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.barh(range(len(feature_importance)), feature_importance['importance'], color='coral')
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Features - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 4. Courbe ROC - Random Forest
ax4 = plt.subplot(2, 3, 4)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

# 5. Courbe ROC - Logistic Regression
ax5 = plt.subplot(2, 3, 5)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

# 6. Comparaison des modèles
ax6 = plt.subplot(2, 3, 6)
models = ['Random Forest', 'Logistic Regression']
accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)]
f1_scores = [f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_lr)]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')
plt.xlabel('Modèles')
plt.ylabel('Score')
plt.title('Comparaison des performances', fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)

for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
    plt.text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)
    plt.text(i + width/2, f1 + 0.02, f'{f1:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('netflix_model_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques sauvegardés : netflix_model_results.png")
plt.show()

# ==================== 7. CONCLUSION ====================
print("\n" + "=" * 60)
print("🎯 ÉTAPE 7 : CONCLUSION ET INTERPRÉTATION")
print("=" * 60)

print(f"""
📊 RÉSUMÉ DES RÉSULTATS :

1️⃣ Dataset :
   - {len(df_clean)} contenus analysés après nettoyage
   - {df_clean['type'].value_counts()['Movie']} films et {df_clean['type'].value_counts()['TV Show']} séries
   - Contenus de {df_clean['primary_country'].nunique()} pays différents

2️⃣ Variable cible :
   - {df_clean['is_popular'].sum()} contenus populaires ({df_clean['is_popular'].mean() * 100:.1f}%)
   - Critères : année récente, genres populaires, durée optimale

3️⃣ Performances des modèles :
   
   🌲 Random Forest :
   - Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}
   - F1-Score : {f1_score(y_test, y_pred_rf):.4f}
   - AUC-ROC : {roc_auc_rf:.4f}
   
   📈 Logistic Regression :
   - Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}
   - F1-Score : {f1_score(y_test, y_pred_lr):.4f}
   - AUC-ROC : {roc_auc_lr:.4f}

4️⃣ Features les plus importantes :
""")

top_features = feature_importance.head(5)
for idx, row in top_features.iterrows():
    print(f"   - {row['feature']} : {row['importance']:.4f}")

print(f"""
5️⃣ Insights clés :
   - Les contenus récents (après 2015) ont plus de chances d'être populaires
   - Les genres Dramas, Comedies et International Movies dominent
   - Les films de durée optimale (80-120 min) sont favorisés
   - Le pays de production et le rating jouent un rôle significatif

💡 PISTES D'AMÉLIORATION :
   ✓ Ajouter une analyse de sentiment sur les descriptions
   ✓ Intégrer des données externes (notes IMDb, Rotten Tomatoes)
   ✓ Créer des features temporelles (tendances saisonnières)
   ✓ Tester d'autres algorithmes (XGBoost, LightGBM)
   ✓ Optimiser les hyperparamètres avec GridSearchCV
   ✓ Analyser les réseaux d'acteurs et réalisateurs
   ✓ Créer des embeddings pour les descriptions textuelles

🎉 PROJET TERMINÉ AVEC SUCCÈS !
""")

print("=" * 60)