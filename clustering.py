import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME
# ─────────────────────────────────────────────
df = pd.read_csv('marketing_campaign.csv', sep='\t')
print(f"Ham veri: {df.shape[0]} satır, {df.shape[1]} sütun")

# ─────────────────────────────────────────────
# 2. VERİ ÖN İŞLEME
# ─────────────────────────────────────────────

# Gereksiz sütunları düşür
df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], inplace=True)

# Aykırı doğum yıllarını çıkar
df = df[df['Year_Birth'] >= 1940].copy()

# Anlamsız medeni hal değerlerini birleştir
df['Marital_Status'] = df['Marital_Status'].replace({'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'})

# Aykırı gelir değerini çıkar
df = df[df['Income'] < 200000].copy()

# Eksik gelir değerlerini KNN ile doldur
imputer = KNNImputer(n_neighbors=5)
df['Income'] = imputer.fit_transform(df[['Income']])

# Yeni özellikler türet
df['Age'] = 2015 - df['Year_Birth']
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
df['Seniority'] = (pd.Timestamp('2015-01-01') - df['Dt_Customer']).dt.days
df['Total_Spent'] = df[['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
df['Total_Purchases'] = df[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']].sum(axis=1)
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
df['Total_AcceptedCmp'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
df['Spend_Rate'] = df['Total_Spent'] / (df['Income'] + 1)

# Eğitimi sayıya çevir
edu_map = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
df['Education_Enc'] = df['Education'].map(edu_map)

# Medeni hali one-hot encode yap
df = pd.get_dummies(df, columns=['Marital_Status'], drop_first=True)

# Ham sütunları düşür
df.drop(columns=['Year_Birth', 'Dt_Customer', 'Education'], inplace=True)

print(f"Temizleme sonrası: {df.shape[0]} satır")

# ─────────────────────────────────────────────
# 3. ÖLÇEKLENDİRME VE PCA
# ─────────────────────────────────────────────
feature_cols = ['Income', 'Age', 'Seniority', 'Total_Spent', 'Total_Purchases',
                'Total_Children', 'Total_AcceptedCmp', 'Spend_Rate',
                'NumWebVisitsMonth', 'Recency', 'Education_Enc',
                'MntWines', 'MntMeatProducts', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases']
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
X_scaled = RobustScaler().fit_transform(X)

pca = PCA(n_components=0.85, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_scaled.shape[1]} özellik → {X_pca.shape[1]} bileşen ({pca.explained_variance_ratio_.sum():.2%} varyans)")

# ─────────────────────────────────────────────
# 4. OPTİMAL K BULMA
# ─────────────────────────────────────────────
k_range = range(2, 9)
inertias, silhouettes, db_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))
    db_scores.append(davies_bouldin_score(X_pca, labels))

best_k = k_range.start + silhouettes.index(max(silhouettes))
print(f"En iyi K: {best_k}  |  Silhouette: {max(silhouettes):.4f}")

# Elbow grafiği
plt.figure(figsize=(7, 4))
plt.plot(list(k_range), inertias, 'o-', color='#534AB7', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'En iyi k={best_k}')
plt.title('Elbow Yöntemi'); plt.xlabel('K'); plt.ylabel('Inertia')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Silhouette grafiği
plt.figure(figsize=(7, 4))
plt.plot(list(k_range), silhouettes, 's-', color='#1D9E75', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'En iyi k={best_k}')
plt.title('Silhouette Skoru'); plt.xlabel('K'); plt.ylabel('Skor')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Davies-Bouldin grafiği
plt.figure(figsize=(7, 4))
plt.plot(list(k_range), db_scores, '^-', color='#BA7517', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'En iyi k={best_k}')
plt.title('Davies-Bouldin Skoru (↓ iyi)'); plt.xlabel('K'); plt.ylabel('DB Skoru')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 5. 3 ALGORİTMA
# ─────────────────────────────────────────────

# K-Means
km_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_pca)

# Agglomerative
agg_labels = AgglomerativeClustering(n_clusters=best_k, linkage='ward').fit_predict(X_pca)

# DBSCAN - en iyi parametreyi bul
best_sil, best_eps, best_ms = -1, 1.5, 5
for eps in [0.8, 1.0, 1.5, 2.0, 2.5]:
    for ms in [3, 5, 8]:
        lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_pca)
        n_cls = len(set(lbl)) - (1 if -1 in lbl else 0)
        if n_cls >= 2 and (lbl != -1).sum() > 50:
            s = silhouette_score(X_pca[lbl != -1], lbl[lbl != -1])
            if s > best_sil:
                best_sil, best_eps, best_ms = s, eps, ms

db_labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X_pca)
db_n = len(set(db_labels)) - (1 if -1 in db_labels else 0)

# Skorlar
km_sil  = silhouette_score(X_pca, km_labels)
agg_sil = silhouette_score(X_pca, agg_labels)

print(f"\nK-Means            Silhouette: {km_sil:.4f}  |  Küme: {best_k}")
print(f"Agglomerative      Silhouette: {agg_sil:.4f}  |  Küme: {best_k}")
print(f"DBSCAN             Silhouette: {best_sil:.4f}  |  Küme: {db_n}  |  eps={best_eps}, min_samples={best_ms}")

# ─────────────────────────────────────────────
# 6. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────
colors = ['#534AB7', '#1D9E75', '#D85A30', '#BA7517', '#993556']

# K-Means scatter
plt.figure(figsize=(8, 5))
for i in range(best_k):
    mask = km_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=f'Küme {i}', alpha=0.5, s=20)
plt.title(f'K-Means  |  k={best_k}  |  Silhouette={km_sil:.3f}')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()

# Agglomerative scatter
plt.figure(figsize=(8, 5))
for i in range(best_k):
    mask = agg_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=f'Küme {i}', alpha=0.5, s=20)
plt.title(f'Agglomerative (Ward)  |  k={best_k}  |  Silhouette={agg_sil:.3f}')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()

# DBSCAN scatter
plt.figure(figsize=(8, 5))
unique_labels = sorted(set(db_labels))
for i, lbl in enumerate(unique_labels):
    mask = db_labels == lbl
    label_name = f'Gürültü' if lbl == -1 else f'Küme {lbl}'
    color = 'black' if lbl == -1 else colors[i % len(colors)]
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label_name, alpha=0.5, s=20)
plt.title(f'DBSCAN  |  k={db_n}  |  Silhouette={best_sil:.3f}  |  eps={best_eps}, ms={best_ms}')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 7. KÜME PROFİLLERİ (K-Means)
# ─────────────────────────────────────────────
df['Cluster'] = km_labels
profile_cols = ['Income', 'Age', 'Total_Spent', 'Total_Purchases', 'Total_Children', 'Total_AcceptedCmp', 'Recency']
cluster_profile = df.groupby('Cluster')[profile_cols].mean().round(1)
print("\nKüme Profilleri (K-Means):")
print(cluster_profile.to_string())

# Gelir bar grafiği
plt.figure(figsize=(7, 4))
plt.bar(cluster_profile.index.astype(str), cluster_profile['Income'], color=colors[:best_k])
plt.title('Kümelere Göre Ortalama Gelir'); plt.xlabel('Küme'); plt.ylabel('Gelir (€)')
plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

# Harcama bar grafiği
plt.figure(figsize=(7, 4))
plt.bar(cluster_profile.index.astype(str), cluster_profile['Total_Spent'], color=colors[:best_k])
plt.title('Kümelere Göre Toplam Harcama'); plt.xlabel('Küme'); plt.ylabel('Harcama (€)')
plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

# Profil heatmap
plt.figure(figsize=(10, 4))
profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
sns.heatmap(profile_norm.T, annot=cluster_profile.T.round(0), fmt='.0f',
            cmap='RdYlGn', linewidths=0.5, annot_kws={'size': 9})
plt.title('Küme Profil Haritası'); plt.xlabel('Küme'); plt.tight_layout(); plt.show()


