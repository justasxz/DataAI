import os
# Nustatome Keras 3 backend'ą į PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras

import torch
from transformers import PatchTSTForPrediction

# 1. Užkrauname TIKRĄJĮ iš anksto apmokytą modelį iš Hugging Face
model_name = "ibm-granite/granite-timeseries-patchtst"
print(f"Kraunamas modelis: {model_name}...")
model = PatchTSTForPrediction.from_pretrained(model_name)
model.eval() # Perjungiame į spėjimų (inference) režimą

# 2. Pasiruošiame duomenis
context_length = model.config.context_length
num_features = model.config.num_input_channels

# Sukuriame atsitiktinį tenzorių demonstracijai
# (Batch dydis = 1, praeities žingsniai = context_length, kintamieji = num_features)
past_values = torch.randn(1, context_length, num_features)
print(past_values)
# 3. Atliekame spėjimą
print("Atliekamas spėjimas...")
with torch.no_grad():
    outputs = model(past_values=past_values)

# 4. Ištraukiame rezultatus
forecast = outputs.prediction_outputs

print("\n--- REZULTATAI ---")
print(f"Spėjimo tenzoriaus forma (Batch, Prediction Length, Features): {forecast.shape}")
print(f"Modelis nuspėjo {forecast.shape[1]} žingsnių į priekį kiekvienam iš {num_features} kintamųjų.")
print("Spėjimo reikšmės (pirmi 5 žingsniai):")
print(forecast[0, :5, 0])