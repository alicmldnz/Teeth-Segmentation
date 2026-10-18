# *Teeth-Segmentation
# *📁 Klasör yapısı: 

teeth_segmentation/ 
├── data/ 
│   ├── imagesTr/ 
│   ├── labelsTr/ 
│   └── dataset.json 
├── src/ 
│   ├── dataset.py 
│   ├── model.py 
│   ├── losses.py 
│   └── train.py 
├── notebooks/ 
│   └── EDA.ipynb 
└── main.py 

 

teeth_segmentation/ 

├── data/               ← veriler 

├── src/                ← model, dataset, loss, train 

├── notebooks/          ← keşifsel analizler, prototipleme 

│   └── EDA.ipynb       ← ilk veri inceleme not defteri 

├── train.py            ← eğitim başlatıcı 

├── main.py             ← her şeyi birleştiren sistem 

 

# ✅ ADIM 2: Dataset’i Hazırla (dataset.py) 

🎯 Amaç: .mha dosyalarını yükleyen, gerekli preprocessing'i yapan, batch-friendly hale getiren bir Dataset sınıfı yazmak. 

Veri türü: 

Giriş: 3D CBCT volumetrik görüntüler 

Maskeler: Çok sınıflı segmentasyon maskeleri 

Hedef: Modelin multi-class 3D segmentasyon yapması 

Şu anda bu Dataset sınıfıyla: 

Giriş ve maske dosyalarını yüklüyoruz ✅ 

Normalize ediyoruz ✅ 

Sabit boyuta getiriyoruz ✅ 

Çok sınıflı segmentasyon yapacak şekilde hazırlıyoruz ✅ 

🔜 Devam: 

ADIM 3: Model dosyasını yazacağız (model.py) 

ADIM 4: Loss fonksiyonu ve metric tanımlayacağız (losses.py) 

ADIM 5: Eğitim ve doğrulama döngüsü (train.py) 

ADIM 6: Ana dosya (main.py) ile her şeyi çalıştıracağız. 

# ✅ ADIM 3: MODEL DOSYASINI YAZMAK (model.py) 

 

*🎯 Açıklamalar:

in_channels=1 

CBCT verisi gri tonlu → 1 kanal 

out_channels=49 

49 sınıflı segmentasyon 

ConvTranspose3D 

Upsampling için 

torch.cat(...) 

Skip connection (encoder'dan decoder'a bilgi aktarıyor) 

final_conv 

Her voxel için sınıf skorları verir 

 

# 🧠 Voxel Nedir? 

Veriler hacimsel (volumetric) olduğu için, her nokta 3 boyutludur 

Bu 3D piksellere voxel (volumetric pixel) denir 
 

# 🔥 Cross Entropy Nedir? 

📌 Kısa tanım: 

Cross entropy, modelin tahmin ettiği dağılım ile gerçek dağılım arasındaki farkı ölçen bir loss fonksiyonudur. 

📊 Tahmin = modelin softmax çıktısı 
 🎯 Gerçek = one-hot vektör veya sınıf ID'si 

 

📚 Biraz Daha Açalım 

Diyelim ki modelin şu 3 sınıfa ait bir örneği tahmin ediyor: 

Sınıf 

Gerçek Etiket (target) 

Model Tahmini (softmax output) 

Arka plan (0) 

0 

0.1 

Diş (1) 

1 ← gerçek sınıf 

0.7 

Kemik (2) 

0 

0.2 

Gerçek sınıf → "Diş" → index 1 
 Model tahmini → Diş için %70 olasılık vermiş 

 

# 🧠 Cross Entropy Loss ne yapar? 

Sadece gerçek sınıfın tahminine bakar, -log(probability) hesaplar: 

Loss = - log(0.7) ≈ 0.356 
 

Bu değer: 

Yüksek güven varsa → düşük loss ✔️ 

Yanlış sınıfa yüksek güven varsa → yüksek loss ❌ 

 

# 📦 Neden Segmentasyona Uygun? 

Senin 3D segmentasyon problemin şunu yapıyor: 

Giriş → CBCT hacmi (1 kanal) 

Çıkış → [B, 49, D, H, W] → her voxel için 49 sınıflık skor 

Gerçek → [B, D, H, W] → her voxel için doğru sınıf ID’si (örneğin 0, 1, 2, ..., 48) 

CrossEntropyLoss, her voxel için bu hesaplamayı yapar ve ortalamasını alır. 

 

✅ Bu yüzden çok sınıflı segmentasyon için ideal: 

Her voxel'e özel sınıf skoru üretildiği için 

Softmax + log işlemini içinde kendi yapar 

One-hot vektör istemez, sadece sınıf ID’leri yeterli 

GPU için optimize edilmiştir → çok hızlı çalışır 

✨ Özet: 

Özellik 

Açıklama 

Loss tipi 

nn.CrossEntropyLoss() 

Girdi (model output) 

[B, C, D, H, W] — sınıf skorları (logits) 

Hedef (ground-truth) 

[B, D, H, W] — sınıf ID’leri (0–48) 

Softmax gerekir mi? 

❌ Hayır, CrossEntropy içinde otomatik yapılır 

Çok sınıflı segmentasyona uygun mu? 

✅ Evet, endüstri standardıdır 

 

"Ama Dice Score veya IoU bizim asıl segmentasyon başarımızı göstermiyor mu? O zaman neden loss fonksiyonu olarak da onları kullanmıyoruz?" 

 

# ✅ Dice Loss ve IoU Loss Nedir? 

İkisi de, tahmin edilen segment ile gerçek segmentin ne kadar çakıştığını ölçen overlap tabanlı metriklerdir. 

📦 Dice Coefficient: 

2 * (|A ∩ B|) / (|A| + |B|) 

📦 IoU (Intersection over Union): 

|A ∩ B| / |A ∪ B| 

İkisi de ölçüm metrikleri olarak harika. Özellikle küçük yapıları segment ederken çok etkilidirler. 
 Ama neden genelde loss fonksiyonu olarak CrossEntropy tercih edilir, onu açıklayalım 👇 

 

🎯 Neden Dice/IoU kullanmıyoruz? (veya tek başına kullanmıyoruz) 

❌ 1. Dice/IoU diferansiyellenmesi zor (özellikle multi-class'ta) 

Loss fonksiyonu, geriye türevlenebilir olmalı (gradient hesaplanmalı). 
 Dice ve IoU bu konuda sıkıntılı: 

IoU çok parçalı türev içerir 

Dice'ın düzleştirilmesi ve çok sınıfa uyarlanması karmaşıktır 

Özellikle class-imbalance varsa kararsız hale gelir 

 

❌ 2. CrossEntropy GPU için optimize edilmiştir, Dice değil 

CrossEntropy, PyTorch’un backend’inde C++/CUDA seviyesinde optimize edilmiştir. 
 Dice/IoU loss genelde Python'da yazılır, yavaş olabilir. 

 

❌ 3. Dice/IoU loss genelde loss landscape'i bozuk olur 

Yani: 

Eğitim sırasında optimizasyon zorlaşabilir 

Özellikle başlangıçta model tamamen yanlış tahmin ediyorsa, Dice = 0 → gradient çok zayıf olur 

 

✅ Peki Ne Yapıyoruz? 

🔁 Genel Pratik Şudur: 

🔥 Eğitim için: 

CrossEntropyLoss kullan 

(İstersen + DiceLoss ile kombine et, ama dikkatli ayarla) 

🔍 Değerlendirme için: 

Dice Score, IoU, Pixel Accuracy gibi ölçüm metriklerini kullan 

Yani “model ne kadar iyi segment etti” sorusunu bunlarla cevapla 

🧠 Özetle: 

Loss Türü 

Kullanım Durumu 

Açıklama 

CrossEntropy 

# ✅ Eğitim için çok ideal 

Stabil, hızlı, GPU dostu 

DiceLoss 

⚠️ Kombinasyonla olabilir 

Tek başına zor, ama tamamlayıcı 

IoU Loss 

🚫 Zor türevlenir, önerilmez 

Genelde metrik olarak kullanılır 

Dice/IoU Skor 

✅ Değerlendirme metrikleri 

Modeli ne kadar iyi segment ettiğini ölçer 

 

# ✅ ADIM 4 – Eğitim Döngüsü (CrossEntropy + Dice Score) 

Bu adımda: 

CrossEntropyLoss ile eğitim yapacağız 

Her epoch’ta Dice Score ile başarıyı ölçeceğiz 

Tüm yapıyı modüler ve temiz yazacağız 

✅ 1. losses.py → CrossEntropyLoss fonksiyonu 

✅ 2. metrics.py → Dice Score metriği 

🎯 Şimdi bu 2 modülü train.py içinde kullanacağız 

# ✅ Özet: Adım 4 tamamlandı! 

Loss -> CrossEntropyLoss() → losses.py ✅ 

Metric-> Dice Score → metrics.py ✅ 

Kullanım-> train.py içinde çağrılır ✅ 

 

train_one_epoch 

Her epoch'ta model eğitilir 

evaluate 

Doğrulama verisinde loss + dice ölçülür 

best_model.pth 

En yüksek dice değerine ulaşılan model kaydedilir 

%80 train / %20 val 

Otomatik ayrım yapılır 

 

 

 

 

# ✅ Class-weighted CrossEntropyLoss 

Segmentasyon maskelerinde bazı sınıflar (örneğin background veya mandibula) aşırı büyük, bazıları ise çok küçük (örneğin kanal, diş kökü). 

Bu dengesizlik nedeniyle model yalnızca büyük sınıfları öğrenme eğilimindeydi. 

compute_class_weights_from_labels() fonksiyonu ile: 

Background (0), çok baskın sınıflar (1, 7) ve voxel sayısı 0 olan sınıflar tamamen dışlandı 

Kalan sınıflar için voxel sayısının tersine dayalı, power + min-max normalize edilmiş class_weights üretildi. 

Bu weights, CrossEntropyLoss(weight=..., ignore_index=255) yapısıyla modele verildi. 

Eğitimde label_batch içinde sınıf 0, 1, 7 → 255 olarak işaretlenerek loss hesabından çıkarıldı. 

# ✅ EarlyStopping 

Modelin validation dice skorunda gelişme gözlenmediğinde eğitim boş yere uzamasın diye entegre edildi. 

patience=10 kullanıldı: 10 epoch boyunca iyileşme yoksa eğitim durur. 

early_stopper(val_dice) çağrısı epoch sonunda kontrol edilir. 

# ✅ Mixed Precision Training (autocast + GradScaler) 

Eğitim süresi kısalsın, GPU belleği daha verimli kullanılsın diye eklendi. 

autocast("cuda") ile modelin forward pass kısmı FP16 olarak çalışır 

GradScaler ile loss-backward işlemi güvenli şekilde yapılır 

Özellikle büyük 3D verilerle çalışırken çok ciddi hız ve bellek avantajı sağlar. 

# ✅ Eğitim Seti Alt Kümesi ile Test 

Eğitim süresi çok uzun olduğu için veri setinin %20'si ile eğitim, %5'i ile validasyon yapıldı. 

Subset() ile kontrollü bir şekilde veri ayrımı yapıldı. 

Bu yöntem sayesinde ağırlık testleri ve tahmin kalitesi hızlıca test edilebildi. 

# ✅ Dice Score Detaylandırması 

metrics.py içindeki dice_score() fonksiyonu hem train hem validation aşamasında kullanıldı. 

Sadece foreground sınıflar için dice hesaplandı (class 0 hariç). 

Eğitim sırasında modelin ne kadar iyi segmentasyon yaptığı izlenebildi. 

# ✅ Inference & Görselleştirme 

inference.py: best_model.pth ağırlıkları ile tahmin alıp .mha maskesi üretir. 

visualize_mask.py: üretilen tahmin maskelerini matplotlib ile slice bazlı görselleştirir. 

 

 

 