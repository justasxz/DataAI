import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Backend nustatomas prieš importuojant keras
os.environ["KERAS_BACKEND"] = "torch"
import keras

# 1. Duomenų užkrovimas
titanic = pd.read_csv(r'data/titanic_train.csv')
titanic_test = pd.read_csv(r'data/titanic_test.csv')

# 2. Užpildome amžiaus spragas (naudojame train duomenis abiem atvejais, kad išvengtume duomenų nutekėjimo)
age_map = titanic.groupby(['Pclass', 'Sex'])['Age'].transform('mean')
titanic['Age'] = titanic['Age'].fillna(age_map)
titanic_test['Age'] = titanic_test['Age'].fillna(age_map)

# 3. Indeksų nustatymas (SVARBU: PassengerId turi likti, kol jį nustatom kaip indeksą)
titanic.set_index("PassengerId", inplace=True)
titanic_test.set_index("PassengerId", inplace=True)

# 4. Stulpelių suvienodinimas (SVARBU: modelis turi gauti tuos pačius stulpelius ta pačia tvarka)
# Išsaugome 'Survived' tikslui
y = titanic['Survived']

# Paliekame tik skaitinius stulpelius abiejuose rinkiniuose
titanic = titanic.select_dtypes(include=['int64', 'float64']).drop(columns=['Survived'], errors='ignore')
titanic_test = titanic_test.select_dtypes(include=['int64', 'float64'])

# 5. Papildomų tuščių reikšmių valymas (pvz., Fare stulpelyje teste būna NaN)
titanic = titanic.fillna(0)
titanic_test = titanic_test.fillna(0) # Priskyrimas būtinas: titanic_test = ...

# Užtikriname, kad test rinkinys turi tuos pačius stulpelius kaip train
X = titanic.copy()
X_test_raw = titanic_test.copy()

# 6. Skalavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_raw)

# 7. Modelio kūrimas
model = keras.Sequential([
    keras.layers.Input(shape=(X_scaled.shape[1],)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dropout(0.1), # nuresetinti keleta neuronų svorių atsitiktinai į nulį
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
adam_op = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=adam_op, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
es = keras.callbacks.EarlyStopping(patience=13,restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3, min_lr=0.0001)
checkpoint = keras.callbacks.ModelCheckpoint(filepath="models/best_model.keras",save_best_only=True)
# 8. Apmokymas
model.fit(X_scaled, y, batch_size=8, epochs=500, validation_split=0.15, verbose=1, callbacks=[es,rlr,checkpoint])
# model = keras.saving.load_model(r"models/best_model.keras")
# 9. Prognozės
y_pred_prob = model.predict(X_test_scaled).flatten()
# Konvertuojame tikimybes į 0 arba 1 (sigmoid grąžina reikšmes nuo 0 iki 1)
y_pred = (y_pred_prob > 0.5).astype(int)

# 10. Rezultatų išsaugojimas
result_df = pd.DataFrame({'PassengerId': titanic_test.index, 'Survived': y_pred})
result_df.to_csv('titanic_predictions.csv', index=False)

print("\nPrognozės sėkmingai išsaugotos į 'titanic_predictions.csv'!")
print("Naudotas backend:", keras.backend.backend())

# import random

# ats = random.randint(0,10)

# # ivestis = int(input("Iveskite savo spejima"))

# for i in range(0,11):

#     if i == ats:
#         print("Atspejote")
#         print(f"skaicius yra: {i}")
#         break
#     else:
#         print("Deja neatspejote")