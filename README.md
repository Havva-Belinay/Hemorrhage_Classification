# 🧠 Head CT Hemorrhage Detection (Beyin BT Kanama Tespiti)

Bu proje, Beyin Bilgisayarlı Tomografi (BT / Head CT) kesitlerini analiz ederek **Normal** ve **Hemorrhage (Kanama)** durumlarını tespit eden bir derin öğrenme sınıflandırma sistemidir. Proje kapsamında sıfırdan geliştirilen bir CNN mimarisi ile modern bir aktarım öğrenimi (Transfer Learning) modeli olan **ConvNeXt-Tiny** eğitilmiş, performansları karşılaştırılmış ve kullanıcı etkileşimi için **Gradio** tabanlı modern bir web arayüzü sunulmuştur.

---

## 📌 Özellikler

- **İki Farklı Mimari:**
  - **Custom CNN:** Kaiming normal ağırlık ilklendirmeli, 4 evrişim bloklu ve çift dropout katmanlı özel VGG benzeri mimari.
  - **ConvNeXt-Tiny:** ImageNet ağırlıkları ile önceden eğitilmiş, katman dondurma (fine-tuning) ve özel sınıflandırma başlığı eklenmiş modern mimari.
- **Güvenilir Veri Ayrımı:** Veri sızıntısını (data leakage) engellemek adına stratified (katmanlı) %70 Eğitim / %15 Doğrulama / %15 Test ayrımı.
- **Eğitim Regülarizasyonu:** Erken durdurma (EarlyStopping), dinamik öğrenme oranı indirgeme (`ReduceLROnPlateau`), gradient clipping ve veri çeşitlendirme (Data Augmentation).
- **Etkileşimli Arayüz:** Yüklenen BT görüntüsünü gerçek zamanlı işleyen, güven skorlarını ve sınıflandırma çubuklarını görselleştiren web arayüzü.

---

## 📂 Proje Yapısı

```text
├── head_ct/                         # Görüntü veri seti klasörü
├── labels.csv                       # Veri kümesi etiketleri
├── colab_train.py                   # Model tanımları, eğitim ve değerlendirme hattı
├── colab_interface.py               # Gradio tabanlı test/çıkarım arayüzü
├── hemorrhage_detection_results/    # Kaydedilen ağırlıklar, grafikler ve metrikler
│   ├── custom_cnn_best.pth
│   ├── convnext_best.pth
│   ├── custom_cnn_history.png
│   ├── convnext_history.png
│   ├── Custom_CNN_confusion_matrix.png
│   ├── ConvNeXt_Tiny_confusion_matrix.png
│   └── results.json
└── README.md
