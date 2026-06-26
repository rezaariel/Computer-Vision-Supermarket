import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image

# 1. Tentukan folder sumber data mentah
RAW_DIR = 'cv supermarket/dataset'

# 2. Ambil semua file yang ada di dalam folder
all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Total ada {len(all_files)} gambar yang siap disortir.")

for all_file in all_files:
    img_path = os.path.join(RAW_DIR, all_file)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.pause(0.1)
    category = input(f'\nKategori untuk {all_file}: ')
    fldr = os.path.join(RAW_DIR, category)
    os.makedirs(fldr, exist_ok=True)
    shutil.move(img_path, fldr)
    plt.close()

print('semua gambar telah disortir')
