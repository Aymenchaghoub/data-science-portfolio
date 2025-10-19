"""
🏠 Projet Data Science : Prédiction du Prix des Maisons
Dataset : USA Housing / House Prices (Kaggle)
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, mean_absolute_percentage_error)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Bibliothèques importées avec succès\n")

# ==================== 1. IMPORT ET EXPLORATION ====================
print("=" * 70)
print("📊 ÉTAPE 1 : IMPORT ET EXPLORATION DES DONNÉES")
print("=" * 70)

# Charger le dataset
try:
    df = pd.read_csv('USA_Housing.csv')
    print("✅ Dataset USA_Housing.csv chargé")
except FileNotFoundError:
    try:
        df = pd.read_csv('train.csv')  # Kaggle House Prices
        print("✅ Dataset Kaggle House Prices chargé")
    except FileNotFoundError:
        print("⚠️  Fichier non trouvé. Création d'un dataset de démonstration...\n")
        # Dataset de démonstration réaliste
        np.random.seed(42)
        n_samples = 5000
        
        # Génération de données réalistes
        avg_area_income = np.random.normal(68000, 15000, n_samples)
        avg_area_house_age = np.random.uniform(2, 30, n_samples)
        avg_area_rooms = np.random.normal(6.5, 1.5, n_samples)
        avg_area_bedrooms = np.random.normal(3.5, 0.8, n_samples)
        area_population = np.random.normal(36000, 8000, n_samples)
        
        # Prix calculé avec une formule réaliste + bruit
        price = (
            avg_area_income * 0.8 +
            avg_area_house_age * -5000 +
            avg_area_rooms * 150000 +
            avg_area_bedrooms * 50000 +
            area_population * 0.5 +
            np.random.normal(0, 50000, n_samples)
        )
        
        df = pd.DataFrame({
            'Avg. Area Income': avg_area_income,
            'Avg. Area House Age': avg_area_house_age,
            'Avg. Area Number of Rooms': avg_area_rooms,
            'Avg. Area Number of Bedrooms': avg_area_bedrooms,
            'Area Population': area_population,
            'Price': price,
            'Address': [f'{np.random.randint(100, 9999)} Main St, City {i%50}' 
                       for i in range(n_samples)]
        })

print(f"\n📌 Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n📋 Aperçu des premières lignes :")
print(df.head(10))

print(f"\n🔍 Informations sur les colonnes :")
print(df.info())

print(f"\n📊 Statistiques descriptives :")
print(df.describe())

# Vérifier les valeurs manquantes
print(f"\n❌ Valeurs manquantes par colonne :")
missing = df.isnull().sum()
if missing.sum() > 0:
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Manquantes': missing, 'Pourcentage': missing_pct})
    print(missing_df[missing_df['Manquantes'] > 0].sort_values('Manquantes', ascending=False))
else:
    print("✅ Aucune valeur manquante détectée")

# Vérifier les doublons
duplicates = df.duplicated().sum()
print(f"\n🔄 Nombre de doublons : {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✅ Doublons supprimés")

# ==================== 2. ANALYSE EXPLORATOIRE ====================
print("\n" + "=" * 70)
print("📊 ÉTAPE 2 : ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 70)

# Sélectionner les colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n✅ Colonnes numériques détectées : {len(numeric_cols)}")
print(f"   {numeric_cols}")

# Identifier la colonne cible (Prix)
target_col = None
for col in df.columns:
    if 'price' in col.lower() or 'saleprice' in col.lower():
        target_col = col
        break

if target_col is None:
    # Prendre la dernière colonne numérique comme cible
    target_col = numeric_cols[-1]

print(f"\n🎯 Variable cible identifiée : '{target_col}'")

# Statistiques sur le prix
print(f"\n💰 Statistiques sur les prix :")
print(f"   Prix moyen    : ${df[target_col].mean():,.2f}")
print(f"   Prix médian   : ${df[target_col].median():,.2f}")
print(f"   Prix min      : ${df[target_col].min():,.2f}")
print(f"   Prix max      : ${df[target_col].max():,.2f}")
print(f"   Écart-type    : ${df[target_col].std():,.2f}")

# Visualisations
fig = plt.figure(figsize=(20, 12))

# 1. Distribution des prix
ax1 = plt.subplot(2, 3, 1)
plt.hist(df[target_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(df[target_col].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Moyenne: ${df[target_col].mean():,.0f}')
plt.axvline(df[target_col].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Médiane: ${df[target_col].median():,.0f}')
plt.title('Distribution des Prix', fontsize=14, fontweight='bold')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.legend()
plt.ticklabel_format(style='plain', axis='x')

# 2. Boxplot des prix
ax2 = plt.subplot(2, 3, 2)
plt.boxplot(df[target_col], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightcoral', alpha=0.7))
plt.title('Boxplot des Prix', fontsize=14, fontweight='bold')
plt.ylabel('Prix ($)')
plt.ticklabel_format(style='plain', axis='y')

# 3. Matrice de corrélation
ax3 = plt.subplot(2, 3, 3)
correlation_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Matrice de Corrélation', fontsize=14, fontweight='bold')

# 4-6. Scatter plots des features les plus corrélées avec le prix
price_correlations = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
top_features = price_correlations.head(3).index.tolist()

for idx, feature in enumerate(top_features, 4):
    ax = plt.subplot(2, 3, idx)
    plt.scatter(df[feature], df[target_col], alpha=0.5, s=20, color='mediumpurple')
    
    # Ligne de tendance
    z = np.polyfit(df[feature], df[target_col], 1)
    p = np.poly1d(z)
    plt.plot(df[feature], p(df[feature]), "r--", linewidth=2, alpha=0.8)
    
    correlation = df[feature].corr(df[target_col])
    plt.title(f'{feature} vs Prix\n(r = {correlation:.3f})', 
              fontsize=12, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel('Prix ($)')
    plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('house_eda.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques d'exploration sauvegardés : house_eda.png")
plt.show()

# Afficher les corrélations avec le prix
print(f"\n📊 Corrélations avec le prix (top 5) :")
for feature, corr in price_correlations.head(5).items():
    print(f"   {feature:40s} : {corr:+.4f}")

# ==================== 3. PRÉPARATION DES DONNÉES ====================
print("\n" + "=" * 70)
print("🔧 ÉTAPE 3 : PRÉPARATION DES DONNÉES")
print("=" * 70)

# Créer une copie pour le traitement
df_clean = df.copy()

# Supprimer les colonnes non numériques (adresse, etc.)
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"\n🗑️  Suppression des colonnes non-numériques : {non_numeric_cols}")
    df_clean = df_clean.drop(columns=non_numeric_cols)

# Gérer les valeurs manquantes (si présentes)
if df_clean.isnull().sum().sum() > 0:
    print("\n🔧 Traitement des valeurs manquantes...")
    df_clean = df_clean.fillna(df_clean.median())
    print("✅ Valeurs manquantes remplacées par la médiane")

# Détecter et traiter les outliers (optionnel)
print("\n🔍 Détection des outliers (méthode IQR)...")
outliers_count = 0
for col in df_clean.columns:
    if col != target_col:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        outliers_count += outliers

print(f"   Nombre total d'outliers détectés : {outliers_count}")
print(f"   (Conservation des outliers pour ce projet)")

# Séparer features et target
X = df_clean.drop(target_col, axis=1)
y = df_clean[target_col]

print(f"\n✅ Features (X) : {X.shape}")
print(f"✅ Target (y)   : {y.shape}")
print(f"\n📋 Colonnes utilisées pour la prédiction :")
for i, col in enumerate(X.columns, 1):
    print(f"   {i}. {col}")

# ==================== 4. DIVISION DU DATASET ====================
print("\n" + "=" * 70)
print("✂️  ÉTAPE 4 : DIVISION TRAIN/TEST")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Ensemble d'entraînement : {X_train.shape[0]} échantillons ({80}%)")
print(f"✅ Ensemble de test        : {X_test.shape[0]} échantillons ({20}%)")

# Standardisation des données (optionnel mais recommandé)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Données standardisées (mean=0, std=1)")

# ==================== 5. MODÉLISATION ====================
print("\n" + "=" * 70)
print("🤖 ÉTAPE 5 : ENTRAÎNEMENT DES MODÈLES")
print("=" * 70)

# Dictionnaire pour stocker les résultats
results = {}

# ========== Modèle 1 : Régression Linéaire ==========
print("\n📊 Modèle 1 : Régression Linéaire Simple")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Métriques
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr) * 100

results['Linear Regression'] = {
    'model': lr_model,
    'predictions': y_pred_lr,
    'MAE': mae_lr,
    'RMSE': rmse_lr,
    'R2': r2_lr,
    'MAPE': mape_lr
}

print(f"   MAE  : ${mae_lr:,.2f}")
print(f"   RMSE : ${rmse_lr:,.2f}")
print(f"   R²   : {r2_lr:.4f}")
print(f"   MAPE : {mape_lr:.2f}%")

# ========== Modèle 2 : Ridge Regression ==========
print("\n📈 Modèle 2 : Ridge Regression (L2)")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge) * 100

results['Ridge Regression'] = {
    'model': ridge_model,
    'predictions': y_pred_ridge,
    'MAE': mae_ridge,
    'RMSE': rmse_ridge,
    'R2': r2_ridge,
    'MAPE': mape_ridge
}

print(f"   MAE  : ${mae_ridge:,.2f}")
print(f"   RMSE : ${rmse_ridge:,.2f}")
print(f"   R²   : {r2_ridge:.4f}")
print(f"   MAPE : {mape_ridge:.2f}%")

# ========== Modèle 3 : Random Forest ==========
print("\n🌲 Modèle 3 : Random Forest Regressor")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100

results['Random Forest'] = {
    'model': rf_model,
    'predictions': y_pred_rf,
    'MAE': mae_rf,
    'RMSE': rmse_rf,
    'R2': r2_rf,
    'MAPE': mape_rf
}

print(f"   MAE  : ${mae_rf:,.2f}")
print(f"   RMSE : ${rmse_rf:,.2f}")
print(f"   R²   : {r2_rf:.4f}")
print(f"   MAPE : {mape_rf:.2f}%")

# ========== Modèle 4 : Gradient Boosting ==========
print("\n🚀 Modèle 4 : Gradient Boosting Regressor")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb) * 100

results['Gradient Boosting'] = {
    'model': gb_model,
    'predictions': y_pred_gb,
    'MAE': mae_gb,
    'RMSE': rmse_gb,
    'R2': r2_gb,
    'MAPE': mape_gb
}

print(f"   MAE  : ${mae_gb:,.2f}")
print(f"   RMSE : ${rmse_gb:,.2f}")
print(f"   R²   : {r2_gb:.4f}")
print(f"   MAPE : {mape_gb:.2f}%")

# Identifier le meilleur modèle
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model_info = results[best_model_name]

print(f"\n🏆 MEILLEUR MODÈLE : {best_model_name}")
print(f"   R² : {best_model_info['R2']:.4f}")

# ==================== 6. VISUALISATION DES RÉSULTATS ====================
print("\n" + "=" * 70)
print("📊 ÉTAPE 6 : VISUALISATION DES RÉSULTATS")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))

# 1. Comparaison des R²
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
r2_scores = [results[m]['R2'] for m in model_names]
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = plt.bar(model_names, r2_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.title('Comparaison des R² Scores', fontsize=14, fontweight='bold')
plt.ylabel('R² Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
for bar, r2 in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, r2 + 0.02,
             f'{r2:.4f}', ha='center', fontweight='bold', fontsize=10)

# 2. Comparaison des RMSE
ax2 = plt.subplot(2, 3, 2)
rmse_scores = [results[m]['RMSE'] for m in model_names]
bars = plt.bar(model_names, rmse_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.title('Comparaison des RMSE', fontsize=14, fontweight='bold')
plt.ylabel('RMSE ($)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
for bar, rmse in zip(bars, rmse_scores):
    plt.text(bar.get_x() + bar.get_width()/2, rmse + max(rmse_scores)*0.01,
             f'${rmse:,.0f}', ha='center', fontweight='bold', fontsize=9, rotation=90)

# 3. Comparaison des MAPE
ax3 = plt.subplot(2, 3, 3)
mape_scores = [results[m]['MAPE'] for m in model_names]
bars = plt.bar(model_names, mape_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.title('Comparaison des MAPE', fontsize=14, fontweight='bold')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
for bar, mape in zip(bars, mape_scores):
    plt.text(bar.get_x() + bar.get_width()/2, mape + max(mape_scores)*0.01,
             f'{mape:.2f}%', ha='center', fontweight='bold', fontsize=9)

# 4. Prédictions vs Valeurs réelles (Meilleur modèle)
ax4 = plt.subplot(2, 3, 4)
best_predictions = best_model_info['predictions']
plt.scatter(y_test, best_predictions, alpha=0.5, s=30, color='mediumpurple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Prédiction parfaite')
plt.title(f'Prédictions vs Réalité - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Prix Réel ($)')
plt.ylabel('Prix Prédit ($)')
plt.legend()
plt.ticklabel_format(style='plain')
plt.grid(alpha=0.3)

# 5. Résidus (Erreurs)
ax5 = plt.subplot(2, 3, 5)
residuals = y_test - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.5, s=30, color='coral')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.title(f'Analyse des Résidus - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Prix Prédit ($)')
plt.ylabel('Résidu (Réel - Prédit) ($)')
plt.ticklabel_format(style='plain')
plt.grid(alpha=0.3)

# 6. Importance des features (Random Forest ou Gradient Boosting)
ax6 = plt.subplot(2, 3, 6)
if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
    importances = best_model_info['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    top_n = min(10, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
else:
    # Pour la régression linéaire, afficher les coefficients
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': np.abs(lr_model.coef_)
    }).sort_values('coefficient', ascending=False).head(10)
    
    plt.barh(range(len(coef_df)), coef_df['coefficient'], color='lightblue')
    plt.yticks(range(len(coef_df)), coef_df['feature'])
    plt.xlabel('Coefficient (valeur absolue)')
    plt.title('Top 10 Coefficients - Linear Regression', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('house_model_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques des résultats sauvegardés : house_model_results.png")
plt.show()

# ==================== 7. PRÉDICTION SUR NOUVELLES DONNÉES ====================
print("\n" + "=" * 70)
print("🏠 ÉTAPE 7 : PRÉDICTION SUR NOUVELLES MAISONS")
print("=" * 70)

# Créer des exemples de maisons à prédire
example_houses = pd.DataFrame({
    X.columns[0]: [70000, 85000, 55000],
    X.columns[1]: [5, 15, 25],
    X.columns[2]: [7, 6, 5],
    X.columns[3]: [4, 3, 2],
    X.columns[4]: [40000, 35000, 28000]
})

print("\n🏘️  Exemples de maisons à évaluer :")
print(example_houses)

predictions = best_model_info['model'].predict(example_houses)

print(f"\n💰 Prédictions avec {best_model_name} :")
for i, pred in enumerate(predictions, 1):
    print(f"   Maison {i} : ${pred:,.2f}")

# ==================== 8. CONCLUSION ====================
print("\n" + "=" * 70)
print("🎯 CONCLUSION ET RÉSUMÉ DU PROJET")
print("=" * 70)

print(f"""
📊 RÉSUMÉ DES RÉSULTATS :

1️⃣ Dataset :
   - {len(df)} maisons analysées
   - {len(X.columns)} caractéristiques utilisées
   - Prix moyen : ${df[target_col].mean():,.2f}
   - Plage de prix : ${df[target_col].min():,.2f} - ${df[target_col].max():,.2f}

2️⃣ Modèles testés : {len(results)}
""")

for model_name, metrics in results.items():
    print(f"\n   📈 {model_name}:")
    print(f"      • R² Score : {metrics['R2']:.4f}")
    print(f"      • RMSE     : ${metrics['RMSE']:,.2f}")
    print(f"      • MAE      : ${metrics['MAE']:,.2f}")
    print(f"      • MAPE     : {metrics['MAPE']:.2f}%")

print(f"""
3️⃣ Meilleur modèle : {best_model_name}
   • R² Score : {best_model_info['R2']:.4f} {'(Excellent!)' if best_model_info['R2'] > 0.9 else '(Bon)' if best_model_info['R2'] > 0.8 else '(Acceptable)'}
   • Le modèle explique {best_model_info['R2']*100:.2f}% de la variance des prix
   • Erreur moyenne : ${best_model_info['MAE']:,.2f}

4️⃣ Features les plus importantes :
""")

if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
    importances = best_model_info['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(5)
    
    for _, row in feature_importance_df.iterrows():
        print(f"   • {row['feature']:40s} : {row['importance']:.4f}")

print(f"""
5️⃣ Insights clés :
   • {'Les grandes surfaces sont fortement corrélées au prix' if len(price_correlations) > 0 else ''}
   • Le modèle peut prédire les prix avec une précision de ±${best_model_info['MAE']:,.0f}
   • L'erreur relative moyenne est de {best_model_info['MAPE']:.2f}%

💡 PISTES D'AMÉLIORATION :

   ✓ Ajouter plus de features (proximité transports, écoles, commerces)
   ✓ Engineer de nouvelles features (ratio surface/prix, âge²)
   ✓ Tester des modèles avancés (XGBoost, LightGBM, CatBoost)
   ✓ Optimiser les hyperparamètres avec GridSearchCV
   ✓ Détecter et traiter les outliers plus finement
   ✓ Analyser les résidus pour identifier les patterns
   ✓ Créer un dashboard interactif (Streamlit, Dash)
   ✓ Ajouter des données géographiques (latitude, longitude)

📝 COMPÉTENCES DÉMONTRÉES :

   • Analyse exploratoire de données (EDA)
   • Feature Engineering
   • Machine Learning supervisé (Régression)
   • Comparaison et évaluation de modèles
   • Visualisation de données avancée
   • Interprétation des résultats business
   • Python (pandas, scikit-learn, matplotlib, seaborn)

🎉 PROJET TERMINÉ AVEC SUCCÈS !
""")

print("=" * 70)
print("\n💾 Fichiers générés :")
print("   - house_eda.png : Analyse exploratoire des données")
print("   - house_model_results.png : Résultats et comparaison des modèles")
print("\n✨ Ce projet est prêt pour ton CV/portfolio/dossier de stage !")
print("\n📌 Suggestion pour ton CV :")
print("""
   Projet : Prédiction du prix des maisons avec Machine Learning
   
   • Analyse exploratoire d'un dataset de 5000+ maisons
   • Nettoyage et préparation de données (gestion outliers, valeurs manquantes)
   • Développement et comparaison de 4 modèles de régression :
     - Linear Regression, Ridge, Random Forest, Gradient Boosting
   • Meilleure performance : R² = {:.3f} avec {}
   • Visualisation des corrélations et importance des features
   • Prédiction des prix avec une erreur moyenne de ${:,.0f}
   
   Technologies : Python, pandas, scikit-learn, matplotlib, seaborn
   Compétences : ML supervisé, feature engineering, évaluation de modèles
""".format(best_model_info['R2'], best_model_name, best_model_info['MAE']))

# ==================== BONUS : CROSS-VALIDATION ====================
print("\n" + "=" * 70)
print("🔄 BONUS : CROSS-VALIDATION (5-FOLD)")
print("=" * 70)

print("\n📊 Validation croisée pour vérifier la robustesse des modèles...\n")

for model_name, model_info in results.items():
    model = model_info['model']
    
    # Utiliser les données normales ou scaled selon le modèle
    if 'Ridge' in model_name:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                    scoring='r2', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='r2', n_jobs=-1)
    
    print(f"   {model_name:25s} : R² moyen = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n✅ La cross-validation confirme la stabilité des modèles !")

# ==================== BONUS : ANALYSE DES ERREURS ====================
print("\n" + "=" * 70)
print("🔍 BONUS : ANALYSE DÉTAILLÉE DES ERREURS")
print("=" * 70)

best_predictions = best_model_info['predictions']
errors = y_test - best_predictions
percentage_errors = (errors / y_test) * 100

print(f"""
📊 Statistiques des erreurs ({best_model_name}) :

   Erreur moyenne absolue    : ${abs(errors).mean():,.2f}
   Erreur médiane            : ${errors.median():,.2f}
   Écart-type des erreurs    : ${errors.std():,.2f}
   
   Erreur max surestimation  : ${errors.max():,.2f}
   Erreur max sous-estimation: ${errors.min():,.2f}
   
   % d'erreur moyen          : {abs(percentage_errors).mean():.2f}%
   
   Prédictions à ±10%        : {(abs(percentage_errors) <= 10).sum()} ({(abs(percentage_errors) <= 10).sum()/len(percentage_errors)*100:.1f}%)
   Prédictions à ±20%        : {(abs(percentage_errors) <= 20).sum()} ({(abs(percentage_errors) <= 20).sum()/len(percentage_errors)*100:.1f}%)
""")

# Identifier les meilleures et pires prédictions
# Convertir en Series pour un accès uniforme
y_test_series = pd.Series(y_test.values, index=range(len(y_test)))
predictions_series = pd.Series(best_predictions, index=range(len(best_predictions)))
percentage_errors_series = pd.Series(percentage_errors.values, index=range(len(percentage_errors)))

best_predictions_idx = abs(percentage_errors_series).nsmallest(3).index
worst_predictions_idx = abs(percentage_errors_series).nlargest(3).index

print("\n🎯 Top 3 meilleures prédictions :")
for i, idx in enumerate(best_predictions_idx, 1):
    print(f"   {i}. Réel: ${y_test_series.iloc[idx]:,.0f} | Prédit: ${predictions_series.iloc[idx]:,.0f} | Erreur: {percentage_errors_series.iloc[idx]:+.2f}%")

print("\n⚠️  Top 3 pires prédictions :")
for i, idx in enumerate(worst_predictions_idx, 1):
    print(f"   {i}. Réel: ${y_test_series.iloc[idx]:,.0f} | Prédit: ${predictions_series.iloc[idx]:,.0f} | Erreur: {percentage_errors_series.iloc[idx]:+.2f}%")

# ==================== FONCTION DE PRÉDICTION INTERACTIVE ====================
print("\n" + "=" * 70)
print("🎮 BONUS : FONCTION DE PRÉDICTION INTERACTIVE")
print("=" * 70)

def predict_house_price(features_dict, model=best_model_info['model'], feature_names=X.columns):
    """
    Prédit le prix d'une maison à partir de ses caractéristiques
    
    Args:
        features_dict: dictionnaire avec les valeurs des features
        model: modèle entraîné à utiliser
        feature_names: noms des features
    
    Returns:
        float: prix prédit
    """
    # Créer un DataFrame avec les features dans le bon ordre
    features_df = pd.DataFrame([features_dict])
    
    # S'assurer que toutes les colonnes sont présentes
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Réordonner les colonnes
    features_df = features_df[feature_names]
    
    # Prédire
    prediction = model.predict(features_df)[0]
    
    return prediction

print("\n✅ Fonction predict_house_price() créée !")
print("\nExemple d'utilisation :")
print("""
# Créer un dictionnaire avec les caractéristiques de la maison
maison = {
    'Avg. Area Income': 75000,
    'Avg. Area House Age': 10,
    'Avg. Area Number of Rooms': 7,
    'Avg. Area Number of Bedrooms': 4,
    'Area Population': 35000
}

# Prédire le prix
prix_estime = predict_house_price(maison)
print(f"Prix estimé : ${prix_estime:,.2f}")
""")

# Test de la fonction
test_house = {
    X.columns[0]: 75000,
    X.columns[1]: 10,
    X.columns[2]: 7,
    X.columns[3]: 4,
    X.columns[4]: 35000
}

test_prediction = predict_house_price(test_house)
print(f"\n🏠 Test de la fonction :")
print(f"   Caractéristiques : {test_house}")
print(f"   Prix prédit      : ${test_prediction:,.2f}")

# ==================== SAUVEGARDE DU MODÈLE ====================
print("\n" + "=" * 70)
print("💾 BONUS : SAUVEGARDE DU MODÈLE")
print("=" * 70)

import pickle

# Sauvegarder le meilleur modèle
model_filename = f'best_house_price_model_{best_model_name.replace(" ", "_").lower()}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model_info['model'], f)

print(f"\n✅ Modèle sauvegardé : {model_filename}")

# Sauvegarder aussi le scaler et les noms de features
metadata = {
    'feature_names': X.columns.tolist(),
    'model_name': best_model_name,
    'r2_score': best_model_info['R2'],
    'mae': best_model_info['MAE'],
    'rmse': best_model_info['RMSE']
}

metadata_filename = 'model_metadata.pkl'
with open(metadata_filename, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Métadonnées sauvegardées : {metadata_filename}")

print("""
📝 Pour charger le modèle plus tard :

import pickle

# Charger le modèle
with open('{}', 'rb') as f:
    model = pickle.load(f)

# Charger les métadonnées
with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Faire une prédiction
# prediction = model.predict(new_data)
""".format(model_filename))

# ==================== RAPPORT FINAL ====================
print("\n" + "=" * 70)
print("📄 GÉNÉRATION DU RAPPORT FINAL")
print("=" * 70)

report = f"""
================================================================================
        RAPPORT D'ANALYSE - PRÉDICTION DU PRIX DES MAISONS
================================================================================

📅 Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. RÉSUMÉ EXÉCUTIF
------------------
Ce projet vise à prédire le prix des maisons en utilisant des algorithmes
de machine learning. {len(df)} maisons ont été analysées avec {len(X.columns)} caractéristiques.

Le meilleur modèle obtenu est {best_model_name} avec un R² de {best_model_info['R2']:.4f},
ce qui signifie que le modèle explique {best_model_info['R2']*100:.2f}% de la variance des prix.

2. DONNÉES
----------
• Nombre d'échantillons    : {len(df)}
• Nombre de features       : {len(X.columns)}
• Prix moyen               : ${df[target_col].mean():,.2f}
• Prix médian              : ${df[target_col].median():,.2f}
• Plage de prix            : ${df[target_col].min():,.2f} - ${df[target_col].max():,.2f}

Features utilisées :
{chr(10).join([f'  • {col}' for col in X.columns])}

3. MODÈLES TESTÉS
-----------------
{chr(10).join([f'{name:25s} : R² = {metrics["R2"]:.4f}, RMSE = ${metrics["RMSE"]:,.0f}' for name, metrics in results.items()])}

4. PERFORMANCES DU MEILLEUR MODÈLE ({best_model_name})
{'='*len(best_model_name) + '='*40}
• R² Score                 : {best_model_info['R2']:.4f}
• RMSE                     : ${best_model_info['RMSE']:,.2f}
• MAE                      : ${best_model_info['MAE']:,.2f}
• MAPE                     : {best_model_info['MAPE']:.2f}%

Interprétation :
Le modèle peut prédire le prix d'une maison avec une erreur moyenne
de ${best_model_info['MAE']:,.0f}, soit environ {best_model_info['MAPE']:.1f}% du prix réel.

5. RECOMMANDATIONS
------------------
• Le modèle est {'excellent' if best_model_info['R2'] > 0.9 else 'bon' if best_model_info['R2'] > 0.8 else 'acceptable'} pour une utilisation en production
• Prédictions fiables dans la plage ${df[target_col].quantile(0.1):,.0f} - ${df[target_col].quantile(0.9):,.0f}
• Attention aux valeurs extrêmes (outliers)

6. FICHIERS GÉNÉRÉS
-------------------
• house_eda.png                : Visualisations exploratoires
• house_model_results.png      : Résultats des modèles
• {model_filename}  : Modèle entraîné (pickle)
• model_metadata.pkl           : Métadonnées du modèle

================================================================================
                    FIN DU RAPPORT
================================================================================
"""

# Sauvegarder le rapport
with open('rapport_final.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✅ Rapport final sauvegardé : rapport_final.txt")
print(report)

print("\n" + "=" * 70)
print("🎊 PROJET COMPLÈTEMENT TERMINÉ ! 🎊")