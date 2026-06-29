# Computer Vision Supermarket

Project ini adalah project sederhana untuk mengenali produk supermarket dari gambar menggunakan machine learning.

# ATTENTION

Model yang dibuat sudah tidak over fitting maupun under fitting, tetapi accuracynya masih sangat rendah [berkisar 0.2----] dan sangat rentan salah dalam menebak produk. Saya ingin membuat model ini menjadi model yang bagus tetapi keterbatasan data assets menjadi lebih sulit

# Poin Utama

- Menggunakan dataset gambar produk supermarket
- Melatih model klasifikasi gambar
- Bisa dipakai untuk prediksi produk dari gambar uji

# File Utama

- [train.py](train.py) untuk melatih model
- [test.py](test.py) untuk mencoba prediksi
- [sorter.py](sorter.py) untuk mengatur data gambar
- [dataset/](dataset) untuk menyimpan data gambar
- [model_supermarket.h5](model_supermarket.h5) hasil model yang sudah dilatih

# Sumber Data

Data asset berasal dari Kaggle: [Supermarket Groceries Image Dataset](https://www.kaggle.com/datasets/aashithakanagala/supermarket-groceries-image-dataset)

# Cara Menjalankan

1. Install dependency:

```bash
pip install tensorflow numpy matplotlib pillow
```

2. Jalankan training:

```bash
python train.py
```

3. Jalankan test prediksi:

```bash
python test.py
```

4. Kalau ingin mengatur data gambar:

```bash
python sorter.py
```

# Cara Git Pull Project Ini

Kalau ingin mengambil update project terbaru dari repository, gunakan perintah:

```bash
git pull origin main
```

Kalau branch-nya bukan main, sesuaikan dengan nama branch yang dipakai.
