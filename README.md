# Yüz Tanıma ve Filtreleme Aracı (Face Recognition Filter)

Bu betik, yerel bilgisayarınızdaki bir klasör dolusu fotoğraf içerisinde, belirlediğiniz "**hedef yüz(ler)i**" arayan ve içerisinde hedeflenen kişilerin olduğu fotoğrafları otomatik olarak ayrı bir klasöre kopyalayan gelişmiş bir araçtır.

Hem **tekil** (bir kişiyi arama) hem de **çoklu** (aynı anda birden çok kişiyi veya bir gruptakileri arama) yüz arama yeteneklerine sahiptir. Çok çekirdekli (multiprocessing) işlem gücünü kullanarak büyük klasörleri bile hızlıca tüketir.

## Özellikler

- **Tek Bir Yüz Arama:** Referans bir fotoğraf verirsiniz (örneğin `reference.jpeg`), yazdığınız klasörde sadece o yüze ait kişiyi bulur.
- **Çoklu Yüz Arama:** Toplu bir grup yüzü referans gösterebilirsiniz. Bunun için:
  - İsterseniz referans fotoğrafları barındıran bir klasör gösterebilirsiniz (varsayılan: `grup_photos`).
  - İsterseniz de `ali.jpg, veli.jpg` gibi virgülle ayrılmış yüz fotoğrafları girebilirsiniz.
- **Mantıksal Filtreler (EN AZ BİRİ / HEPSİ):** Çoklu aramalarda hedeflerdeki kişilerin "*en az birinin*" mi yoksa "*hepsinin birden aynı anda*" mı bulunması gerektiğini seçebilirsiniz.
- **Çok Çekirdek (Multiprocessing) Desteği:** İşlemler çekirdeklere bölünerek hızlandırılır.
- **Otomatik Ölçeklendirme (Resize):** Büyük boyuttaki imajları tararken performansı düşürmemek için HD seviyesinde sınırlandırır (opsiyonel max: 1280px).

## Kurulum ve Gereksinimler

Bu sistem özellikle `dlib` ve `face_recognition` kütüphanelerine büyük ölçüde bağımlıdır.

1. **Gereksinimlerin İndirilmesi:**
   Ortamınızı ayarlarken terminalinize gidip `requirements.txt` dosyasındaki kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
   **Not:** `dlib` kurulumu için sisteminizde `cmake` eklentisine ihtiyaç olabilir (Eğer yoksa Mac sistemler için brew yardımıyla: `brew install cmake` kurabilirsiniz).

2. **Conda Ortamı Önerisi:**
   İzole bir ortam (örneğin Python 3.10 tabanlı) kurup aktif etmeniz şiddetle tavsiye edilir. Varsayılan olarak hata alınması dahilinde `conda activate py310` gibi bir ortamı etkinleştirmeniz gerektiği uyarısı alırsınız.

## Kullanım

Aracı çalıştırmak için aşağıdaki komutu terminalinizde koşturun:

```bash
python filter_faces_from_local_folder.py
```

Başladığında araç size adım adım ve etkileşimli olarak ne yapmak istediğinizi soracak:

### Senaryo 1: Tek Bir Kişinin Yüzünü Filtreleme
* Arama Tipi menüsünden **2**'yi seçersiniz.
* Varsa referans yolunu girersiniz (veya Enter tuşuna basarak varsayılan `reference.jpeg` dosyasını gösterirsiniz).
* Taranacak arşiv klasörünüzün yolunu girersiniz (örn: `emlak_konut`).
* Ve son olarak süzülen fotoğrafların atılacağı klasörü yazıp Enter'a basarsınız (örn: `cikis_tekli`).
* Arkanıza yaslanıp aracın işini bitirmesini beklersiniz.

### Senaryo 2: Gruptaki Kişilerden "Herhangi Birinin" Olduğu Fotoğrafları Seçme (Biri)
* Arama Tipi menüsünden **1**'i seçersiniz.
* İçerisinde aradığınız insanların vesikalık/portre fotoğraflarını barındıran bir klasör verirsiniz (veya boş bırakıp direkt `grup_photos` klasörü okunmasını istersiniz).
* Sistem size eşleşme koşulunu sorduğunda sadece enter'a basıp veya `Biri` diyerek geçersiniz.
* Hedef arşivinizdeki fotoğraflardan, bu verdiğiniz referans kişilerden *herhangi birini* kapsayan tüm fotoğraflar yakalanır ve çıkış klasörüne kopyalanır.

### Senaryo 3: Belirli 3 Arkadaş Tarafından "Birlikte Çekilmiş" Fotoğrafları Listeleme (Hepsi)
* Arama Tipi menüsünden **1**'i seçersiniz.
* Üç arkadaşınızın fotoğraf yollarını virgülle tanımlarsınız: `kisi1.jpg, kisi2.jpg, kisi3.jpg`.
* Size eşleşme koşulu sorulduğunda bu kez **`Hepsi`** yazdığınızda sistem artık sadece hedeflenen bu üç kişinin *üçünün de aynı fotoğrafta yan yana veya birlikte* bulunduğu toplu resimleri yakalar. İkisinin olduğu ancak üçüncü kişinin olmadığı fotoğraflar otomatik es geçilir.

---

*Log Ekranı ve Çıktılar*
İşlemler sırasında araç, bulunan eşleşmeleri size anlık olarak yazdırır. Tarama sonlandığında ise ne kadar fotoğrafın tarandığı, kaç tane eşleşme bulunduğu gibi istatistiklerin dökümünü sunar. Hiç dokunulmamış veya sistemdeki herhangi bir hata dolayısıyla atlanmış olan görseller de log'a yansıtılır.
