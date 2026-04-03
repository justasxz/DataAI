import os
# BŪTINA nustatyti backend'ą Prieš importuojant keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers

# 1. PARAMETRAI
IMAGE_SIZE = (224, 224) # Standartinis ResNet dydis
BATCH_SIZE = 32
NUM_CLASSES = 5 # Pakeisk į savo klasių (kategorijų) skaičių
# DATA_DIR = "kelias/iki/tavo/nuotrauku_aplanko" # Pvz., aplanke turi būti sub-aplankai pagal klases

# # 2. DUOMENŲ UŽKROVIMAS
# # Keras 3 automatiškai tvarko duomenis, net jei backend'as yra PyTorch
# print("Kraunami treniravimo duomenys...")
# train_dataset = keras.utils.image_dataset_from_directory(
#     DATA_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
# )

# print("Kraunami validacijos duomenys...")
# val_dataset = keras.utils.image_dataset_from_directory(
#     DATA_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
# )

# 3. MODELIO ARCHITEKTŪROS KŪRIMAS (Nuo nulio)
def build_model(num_classes):
    # Sukuriame įvesties sluoksnį
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Pridedame duomenų augmentaciją (labai svarbu treniruojant nuo nulio!)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Paimame ResNet50 architektūrą
    # weights=None reiškia, kad modelis NEBUS pretrained (inilizuojamas atsitiktiniais svoriais)
    # include_top=False nuima originalų klasifikatorių, kad galėtume pridėti savo
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=None, 
        input_tensor=x
    )
    
    # Pridedame savo klasifikavimo sluoksnius
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.5)(x) # Apsauga nuo overfitting'o
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model(NUM_CLASSES)

# 4. KOMPILIAVIMAS
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. TRENIRAVIMAS
print("Pradedamas treniravimas su PyTorch backend'u...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20 # Nustatyk norimą epochų skaičių
)

# 6. MODELIO IŠSAUGOJIMAS
model.save("mano_resnet_modelis.keras")
print("Modelis išsaugotas!")