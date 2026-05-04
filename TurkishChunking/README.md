# Turkish Chunking Project

Bu proje, Türkçe metinler üzerinde isim öbekleri (NP), eylem öbekleri (VP), zarf öbekleri (ADVP) ve yan tümceleri (Relative Clause, Complement Clause) saptamak için geliştirilmiştir.

## Özellikler
- **Model**: Conditional Random Fields (CRF)
- **Format**: CoNLL (B-I-O tagging)
- **Hiyerarşik İşaretleme**: Outer Chunk, Inner Chunk ve Clause seviyelerinde analiz.
- **Değerlendirme**: Her seviye için F1, Precision, Recall ve Confusion Matrix hesaplanır.

## Kurulum
Gerekli kütüphaneleri yüklemek için:
```bash
pip install sklearn-crfsuite joblib matplotlib seaborn scikit-learn pandas
```

## Kullanım
Eğitim ve Test işlemini başlatmak için:
```bash
python main.py --mode train
python main.py --mode test
```

## Sonuçlar
Sonuçlar ve grafikler `results/` klasörü altında oluşturulur:
- `cm_OUTER.png`: Dış öbek karmaşıklık matrisi.
- `cm_INNER.png`: İç öbek karmaşıklık matrisi.
- `cm_CLAUSE.png`: Tümce yapısı karmaşıklık matrisi.
- `metrics_*.csv`: Performans metrikleri.

## Örnek Çıktı
`python main.py --mode run` komutu ile örnek bir cümlenin analizini görebilirsiniz.
