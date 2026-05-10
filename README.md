# Customer Personality Analysis — Kümeleme Çalışması

## Proje Hakkında
Müşteri kişilik ve davranış verileri kullanılarak benzer özellikteki müşteriler
anlamlı gruplara (segmentlere) ayrılmıştır.

---

## 📂 Veri Seti
- **Kaynak:** Kaggle
- **Link:** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
- **Dosya:** `marketing_campaign.csv`
- **Satır Sayısı:** ~2.240 müşteri
- **Sütun Sayısı:** 29 özellik
- **Görev Türü:** Gözetimsiz Öğrenme (Unsupervised Learning)

---

## 🛠️ Kullanılan Teknolojiler
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

## 🔄 Uygulanan Adımlar

### 1. Veri Keşfi (EDA)
- Veri seti boyutu ve değişken tipleri incelendi
- İstatistiksel özet çıkarıldı
- Eksik veri analizi yapıldı
- Görselleştirmeler oluşturuldu

### 2. Veri Ön İşleme
- Eksik değerler temizlendi
- Gereksiz sütunlar çıkarıldı
- Kategorik değişkenler encode edildi
- StandardScaler ile normalizasyon uygulandı
- PCA ile boyut indirgeme yapıldı

### 3. Modelleme
- **Elbow Method** ile optimal küme sayısı belirlendi
- **K-Means** algoritması uygulandı
- **Hierarchical Clustering** ile karşılaştırma yapıldı

### 4. Küme Analizi
- Her kümenin demografik ve davranışsal özellikleri incelendi
- Kümeler anlamlı müşteri segmentleri olarak yorumlandı

---

## Sonuçlar

| Küme | Profil |
|---|---|
| Küme 0 | Düşük gelirli, az harcayan müşteriler |
| Küme 1 | Yüksek gelirli, çok harcayan sadık müşteriler |
| Küme 2 | Orta gelirli, kampanyaya duyarlı müşteriler |
| Küme 3 | Genç, düşük harcamalı potansiyel müşteriler |

---

## 🔑 Temel Bulgular
- Müşteriler gelir düzeyi ve harcama alışkanlıklarına göre belirgin gruplara ayrılmaktadır
- Yüksek gelirli müşteriler şarap ve et ürünlerine daha fazla harcama yapmaktadır
- Kampanya duyarlılığı müşteri segmentine göre önemli farklılıklar göstermektedir
- Çocuk sahibi olmak harcama davranışını doğrudan etkilemektedir
