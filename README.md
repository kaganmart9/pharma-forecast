# Pharma Sales Forecasting Project

## Objective

Bu proje, ilaç sektöründeki satış verilerini kullanarak **ürün ve marka bazında haftalık satış tahmini (forecast)** yapmayı amaçlar. Model, promosyon, fiyat ve bölgesel değişiklikleri dikkate alır. Sonuçlar, bir **Streamlit Dashboard** üzerinde etkileşimli şekilde gösterilecektir.

## Dataset

Kaynak: *Pharma Sales Data (Kaggle)*  
Dosya: `data/raw/pharma_sales.csv`  
Zaman aralığı: 2014–2019  
Veri sıklığı: Günlük (haftalığa dönüştürülecek)  
Hedef değişken: `Sales_Value` (satış değeri)

## Forecast setup

- Tahmin ufku (forecast horizon): 8 hafta
- Frekans: haftalık (`W-MON`)
- Seviyeler: SKU → Brand → Region → Country

## Yapı

(klasör yapısını yukarıda belirttiğimiz şekilde buraya kopyala)

## Hızlı Başlangıç

1. Python ortamı oluştur.
2. `requirements.txt` ile bağımlılıkları kur.
3. Veriyi `data/raw/pharma_sales.csv` olarak yerleştir.
4. `python -m src.data_prep` komutuyla temel veri temizliği yap.
