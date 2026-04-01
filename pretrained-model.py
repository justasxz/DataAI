import os
# SVARBU: Nustatome Keras 3 backend'ą į PyTorch PRIEŠ importuojant keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# ==========================================
# 1. DUOMENŲ PARUOŠĖJAS (PROCESORIUS)
# ==========================================
# Tai išsprendžia jūsų minėtą duomenų nesutapimo problemą.
# Jis žino, kaip tiksliai apkarpyti, sumažinti ir normalizuoti vaizdą šiam modeliui.
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# ==========================================
# 2. MODELIO UŽKROVIMAS IR KERAS "WRAPPERIS"
# ==========================================
# Užkrauname PyTorch modelį iš Hugging Face
hf_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
# Kadangi norime Keras 3 patirties, sukuriame Keras modelį, 
# kuris savo viduje naudoja Hugging Face PyTorch modelį.
class HuggingFaceKerasModel(keras.Model):
    def __init__(self, hf_model, **kwargs):
        super().__init__(**kwargs)
        self.hf_model = hf_model

    def call(self, inputs):
        # Hugging face modeliai grąžina specialų objektą. 
        # Mums klasifikacijai reikia tik 'logits' (neapdorotų spėjimų).
        outputs = self.hf_model(inputs)
        return outputs.logits

# Sukuriame mūsų Keras modelio instanciją
keras_model = HuggingFaceKerasModel(hf_model)
# ==========================================
# 3. DUOMENŲ GAVIMAS IR APDOROJIMAS
# ==========================================
# Paimkime atsitiktinį katės paveikslėlį iš interneto
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# ČIA ĮVYKSTA MAGIJA: 
# Procesorius paima paprastą nuotrauką ir paverčia ją į PyTorch tenzorių ("pt"),
# kuris tobulai atitinka ResNet-50 reikalavimus.
inputs = processor(images=image, return_tensors="pt")
print(inputs)
pixel_values = inputs["pixel_values"]  # Tai yra mūsų paruoštas X (duomenys)

# ==========================================
# 4. KLASIFIKACIJA SU KERAS 3
# ==========================================
# Dabar galime drąsiai naudoti Keras .predict() metodą
predictions = keras_model.predict(pixel_values)

# Ištraukiame didžiausios tikimybės klasės indeksą
predicted_class_idx = predictions.argmax(axis=-1)[0]

# Hugging face modelyje yra žodynas (id2label), kuris indeksą paverčia žmogui suprantamu tekstu
label = hf_model.config.id2label[predicted_class_idx]
print(f"\nModelio spėjimas: {label}")