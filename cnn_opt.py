import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
# =====================================================================
# 1. DUOMENŲ PARUOŠIMAS IR SKAIDYMAS
# =====================================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizacija ir kanalo dimensijos pridėjimas
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)

# Taisyklingas skaidymas: 80% treniravimui, 20% validacijai.
# random_state užtikrina, kad kiekvieną kartą paleidus kodą, skaidymas bus toks pat.
x_train_opt, x_val_opt, y_train_opt, y_val_opt = train_test_split(
    x_train, y_train, test_size=0.15, random_state=42
)

# =====================================================================
# 2. OPTIMIZACIJOS FUNKCIJA IR HIPERPARAMETRAI
# =====================================================================
def objective(trial):
    # --- GLOBALŪS HIPERPARAMETRAI ---
    # Kodėl relu/swish: 'relu' yra greitas standartas, 'swish' - modernesnė alternatyva, galinti rasti geresnių sprendimų.
    activation = trial.suggest_categorical("activation", ["relu", "swish"])
    
    # Kodėl log=True (1e-4 iki 1e-2): Mokymosi greitis (žingsnis) turi būti testuojamas eksponentiškai (0.0001, 0.001, 0.01), nes mažieji skirtumai daro didžiausią įtaką.
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    
    # Kodėl 32, 64, 128: Mažesnis skaičius (32) padeda modeliui geriau generalizuoti, didesnis (128) greičiau skaičiuojamas vaizdo plokštėje.
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # --- ARCHITEKTŪROS GYLIS ---
    # Kodėl 1-3 sluoksniai: Leidžiame Optunai pačiai nuspręsti, ar šiai problemai reikia seklos, ar gilios architektūros.
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3) # 1 arba arba 3
    num_dense_layers = trial.suggest_int("num_dense_layers", 1, 2)

    # --- MODELIO KONSTRAVIMAS ---
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    
    # Konvoliuciniai sluoksniai (Dinaminis ciklas)
    for i in range(num_conv_layers): # range(1) | range(2) | range(3)
        # Kodėl 16-64: Pakankamas kiekis filtrų išmokti MNIST skaičių formas, bet ne per didelis, kad modelis nebūtų per sunkus.
        filters = trial.suggest_int(f"filters_{i}", 16, 64, step=16)
        
        # Kodėl 3 arba 5: Nelyginiai skaičiai (turi centrą). 3x3 ieško smulkių detalių, 5x5 ieško stambesnių modelių.
        kernel_size = trial.suggest_categorical(f"kernel_size_{i}", [3, 5])
        
        # Kodėl True/False: Batch Normalization stabilizuoja duomenų sklaida tarp sluoksnių, dažnai pagreitina mokymąsi.
        use_batchnorm = trial.suggest_categorical(f"use_batchnorm_{i}", [True, False])
        
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation, padding="same"))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
            
        model.add(layers.MaxPooling2D((2, 2)))
        
    model.add(layers.Flatten())
    
    # Pilnai sujungti (Dense) sluoksniai (Dinaminis ciklas)
    for i in range(num_dense_layers):
        # Kodėl 32-128: Tai tinklo "smegenų" dydis, priimantis galutinį sprendimą. Per didelis skaičius ves prie persimokymo.
        dense_units = trial.suggest_int(f"dense_units_{i}", 32, 128, step=32)
        
        # Kodėl 0.1-0.5: Dropout išjungia dalį neuronų, versdamas kitus išmokti geresnius bruožus. 0.5 reiškia, kad atsitiktinai išjungiama pusė neuronų.
        dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
        
        model.add(layers.Dense(dense_units, activation=activation))
        model.add(layers.Dropout(dropout_rate))
        
    # Išvesties sluoksnis (10 klasių skaičiams 0-9)
    model.add(layers.Dense(10, activation="softmax"))

    # --- KOMPILIAVIMAS IR VERTINIMAS ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    # Treniruojame tik su x_train_opt
    model.fit(x_train_opt, y_train_opt, epochs=3, batch_size=batch_size, verbose=0)

    # Tikriname egzaminą su x_val_opt
    val_loss, val_accuracy = model.evaluate(x_val_opt, y_val_opt, verbose=1) # cross validacija butu geriau

    return val_accuracy

# =====================================================================
# 3. OPTUNA PALEIDIMAS IR REZULTATAI
# =====================================================================
if __name__ == "__main__":
    print("Pradedama Optuna paieška...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, n_jobs=-1) # Bandome 10 skirtingų kombinacijų

    print("\n--- GERIAUSI RASTI PARAMETRAI ---")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # =====================================================================
    # 4. GALUTINIO MODELIO ATKŪRIMAS IR TESTAVIMAS (PREDICT)
    # =====================================================================
    print("\nStatome galutinį modelį su geriausiais parametrais...")
    
    final_model = keras.Sequential()
    final_model.add(layers.Input(shape=(28, 28, 1)))
    
    # Atkuriame išsaugotą Conv sluoksnių struktūrą
    for i in range(best_params["num_conv_layers"]):
        final_model.add(layers.Conv2D(
            best_params[f"filters_{i}"], 
            (best_params[f"kernel_size_{i}"], best_params[f"kernel_size_{i}"]), 
            activation=best_params["activation"], 
            padding="same"
        ))
        if best_params[f"use_batchnorm_{i}"]:
            final_model.add(layers.BatchNormalization())
        final_model.add(layers.MaxPooling2D((2, 2)))
        
    final_model.add(layers.Flatten())
    
    # Atkuriame išsaugotą Dense sluoksnių struktūrą
    for i in range(best_params["num_dense_layers"]):
        final_model.add(layers.Dense(best_params[f"dense_units_{i}"], activation=best_params["activation"]))
        final_model.add(layers.Dropout(best_params[f"dropout_rate_{i}"]))
        
    final_model.add(layers.Dense(10, activation="softmax"))

    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"]), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    # Dabar, kai žinome geriausius parametrus, treniruojame su PILNAIS originaliais duomenimis (x_train)
    print("Treniruojame galutinį modelį...")
    final_model.fit(x_train, y_train, epochs=5, batch_size=best_params["batch_size"], verbose=1)

    # --- REALI PROGNOZĖ (PREDICT) ---
    print("\n--- TESTAVIMAS IR PROGNOZĖ ---")
    test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"Galutinis modelio tikslumas su visiškai naujais duomenimis: {test_acc:.4f}")

    # Paimame vieną konkretų paveikslėlį
    sample_img = x_test[0:1] 
    tikrasis_skaicius = y_test[0]
    
    spejimai = final_model.predict(sample_img, verbose=0)
    spetas_skaicius = np.argmax(spejimai[0])
    
    print(f"Tikrasis skaičius paveikslėlyje: {tikrasis_skaicius}")
    print(f"Mūsų AI modelis spėja, kad tai: {spetas_skaicius}")