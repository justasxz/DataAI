import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# 1. NUSTATOME KERAS BACKEND Į PYTORCH
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

# -----------------------------------------
# DUOMENŲ PARUOŠIMAS
# -----------------------------------------
print("Kraunami Titaniko duomenys...")
# Naudojame sklearn fetch_openml
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Pasirenkame pagrindinius požymius ir konvertuojame taikinį į int
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = X[features].copy()
y = y.astype(int).values

# Užpildome trūkstamas reikšmes (imputation) ir užkoduojame lytį
X['sex'] = LabelEncoder().fit_transform(X['sex'].astype(str))
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Skalizuojame duomenis
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Padaliname duomenis į Train, Validation (Meta-modeliui treniruoti) ir Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# LSTM modeliui reikalinga 3D forma: (batch_size, timesteps, features)
# Mes naudosime 1 timestep'ą
X_train_lstm = np.expand_dims(X_train, axis=1)
X_val_lstm = np.expand_dims(X_val, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

# -----------------------------------------
# BAZINIŲ MODELIŲ SUKŪRIMAS
# -----------------------------------------

# Modelis 1: Standartinis ANN (Dense)
def build_ann(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs) # x=  dense + iputs
    x = layers.BatchNormalization()(x) # x = batchnorm + x (dense + input) # is esmes vel standartizuojame, kad modelis galetu greiciau konverguoti
    x = layers.Dropout(0.2)(x) # x = dropout + x (dens + input)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs, name="ANN_Base")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Modelis 2: LSTM
def build_lstm(input_shape):
    inputs = keras.Input(shape=input_shape)
    # Kadangi turime tik 1 timestep'ą, LSTM tiesiog interpretuos požymius
    x = layers.LSTM(32, activation='tanh')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs, name="LSTM_Base")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------
# STACKING PROCESAS
# -----------------------------------------

print("\nTreniruojamas ANN modelis...")
ann_model = build_ann(X_train.shape[1])
ann_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

print("Treniruojamas LSTM modelis...")
lstm_model = build_lstm((1, X_train.shape[1]))
lstm_model.fit(X_train_lstm, y_train, epochs=30, batch_size=32, verbose=0)

# Sugeneruojame prognozes validacijos aibei (Tai bus nauji požymiai meta-modeliui)
ann_val_preds = ann_model.predict(X_val, verbose=0)
lstm_val_preds = lstm_model.predict(X_val_lstm, verbose=0)

# Sujungiame prognozes į vieną matricą
stacked_val_features = np.column_stack((ann_val_preds, lstm_val_preds))

# Meta-modelis (paprastas logistinės regresijos atitikmuo Keras aplinkoje)
print("\nTreniruojamas Meta-modelis (Stacking)...")
meta_inputs = keras.Input(shape=(2,)) # 2 įvestys: ANN prognozė ir LSTM prognozė
meta_outputs = layers.Dense(1, activation='sigmoid', name="Meta_Output")(meta_inputs)
meta_model = keras.Model(meta_inputs, meta_outputs, name="Meta_Model")
meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treniruojame meta-modelį ant validacijos prognozių
meta_model.fit(stacked_val_features, y_val, epochs=50, batch_size=16, verbose=0)

# -----------------------------------------
# GALUTINIS TESTAVIMAS
# -----------------------------------------

# 1. Gauname bazinių modelių prognozes testinei aibei
ann_test_preds = ann_model.predict(X_test, verbose=0)
lstm_test_preds = lstm_model.predict(X_test_lstm, verbose=0)

# 2. Sujungiame jas
stacked_test_features = np.column_stack((ann_test_preds, lstm_test_preds))

# 3. Meta-modelis padaro galutinį sprendimą
final_predictions = meta_model.predict(stacked_test_features, verbose=0)
final_predictions_classes = (final_predictions > 0.5).astype(int)

# 4. Palyginame rezultatus
print("\n--- REZULTATAI ---")
print(f"ANN tikslumas (Test): {accuracy_score(y_test, (ann_test_preds > 0.5).astype(int)):.4f}")
print(f"LSTM tikslumas (Test): {accuracy_score(y_test, (lstm_test_preds > 0.5).astype(int)):.4f}")
print(f"Stacking (Meta-modelio) tikslumas (Test): {accuracy_score(y_test, final_predictions_classes):.4f}")