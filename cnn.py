import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Užkrauname MNIST originaliu būdu
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Konvertuojame vieną paveikslėlį į Pandas DataFrame vizualizacijai
# Tai leidžia pamatyti pikselių intensyvumą kaip skaičius
sample_idx = 0
digit_df = pd.DataFrame(x_train[sample_idx])

print(f"Pirmas skaičius masyve yra: {y_train[sample_idx]}")

plt.figure(figsize=(10, 8))
sns.heatmap(digit_df, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.title(f"Skaičiaus {y_train[sample_idx]} struktūra (Pikselių vertės)")
# plt.show()
# plt.imshow(x_train[sample_idx])
# plt.show()
# # Duomenų paruošimas CNN (Normalizacija)
x_train = x_train.astype("float32") / 255.0 # 
x_test = x_test.astype("float32") / 255.0


# print(x_train)
# Pridedame kanalo dimensiją [Batch, H, W, Channel]
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)), # (aukstis(px),plotis(px),spalvu kanalai)
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), # 32 nuotraukas (neuronu kiekis (nevisai tiesa, bet galima)) (3x3 filtras (ima po 9 pikselius))
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), # 32 nuotraukas (neuronu kiekis (nevisai tiesa, bet galima)) (3x3 filtras (ima po 9 pikselius))
    # keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
# Apmokymas
# history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)