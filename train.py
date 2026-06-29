import os
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = (180, 180)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    RAW_DIR,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    RAW_DIR,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

num_classes = len(train_ds.class_names) # Otomatis menghitung jumlah kategori produkmu

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),

    layers.RandomFlip('horizontal', input_shape=(180, 180, 3)),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1),


    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print('\nAI Training started')
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=10
)

model.save('model_supermarket.h5')
print('\nAI Training ended, model saved with name "model_supermaket.h5"')

