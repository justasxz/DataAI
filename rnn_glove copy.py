import os
# CRITICAL: Backendą būtina nustatyti prieš importuojant keras!
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# ==========================================
# 1. Duomenų nuskaitymas ir paruošimas
# ==========================================
file_path = "data/war_peace.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Apribojame tekstą, kad sutaupytume laiko/atminties mokymosi tikslais
max_chars = 50000
raw_text = raw_text[:max_chars]

# Išskaidome tekstą į žodžius ir skyrybos ženklus, paverčiame mažosiomis raidėmis
words = re.findall(r'\b\w+\b|[^\w\s]', raw_text.lower()) 

# Sukuriame unikalų žodyną. Pridedame specialius žodžius:
# <pad> - sekomis sulygiuoti (užpildas)
# <unk> - nežinomiems žodžiams, kurių nebus žodyne
vocab = ["<pad>", "<unk>"] + list(dict.fromkeys(words)) 

# Sukuriame žodynus greitam indeksų ir žodžių konvertavimui
word_index = {w: i for i, w in enumerate(vocab)} 
inv_vocab = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Paverčiame visą tekstą į skaičių (indeksų) seką
word_seq = [word_index.get(w, 1) for w in words]

# ==========================================
# 2. PyTorch Dataset ir DataLoader
# ==========================================
seq_length = 50 # Kiek praeities žodžių modelis matys, kad nuspėtų sekantį
batch_size = 64 # Kiek sekų apdorojama vieno žingsnio metu

class TextDataset(Dataset):
    def __init__(self, sequence, seq_len):
        self.sequence = sequence
        self.seq_len = seq_len

    def __len__(self):
        # Grąžiname galimų sekų skaičių
        return len(self.sequence) - self.seq_len 

    def __getitem__(self, idx):
        # Input (X): seka nuo dabartinio indekso
        x = self.sequence[idx : idx + self.seq_len] 
        # Target (Y): ta pati seka, tik pasislinkusi per vieną žodį į priekį
        y = self.sequence[idx + 1 : idx + self.seq_len + 1] 
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = TextDataset(word_seq, seq_length)
# drop_last=True garantuoja, kad visi batch'ai bus vienodo dydžio (svarbu kai kuriems modeliams)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ==========================================
# 3. GloVe Vektorių (Embeddings) Įkėlimas
# ==========================================
glove_file = "data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
embeddings_index = {}

print("Skaitomas GloVe failas. Tai gali užtrukti...")

with open(glove_file, "r", encoding="utf-8") as f:
    skipped_amount = 0
    for line_number, line in enumerate(f):
        values = line.split()
        
        # Praleidžiame tuščias eilutes arba eilutes, kurios per trumpos būti vektoriumi
        if len(values) < 10: 
            continue
            
        word = values[0]

        try:
            # Bandome paversti likusias reikšmes į skaičius
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            # Jei randa tekstą (pvz., tašką '.') ten, kur turi būti skaičius - ignoruojame šią eilutę
            # Galite atkomentuoti apatinę eilutę, jei norite matyti, kurios eilutės buvo praleistos:
            # print(f"Klaidinga eilutė praleista (Nr. {line_number}): {word}")
            skipped_amount += 1
            continue
print(skipped_amount)
print(f"Sėkmingai įkelta {len(embeddings_index)} žodžių vektorių.")

# Gauname dimensiją iš pirmo sėkmingai įkelto vektoriaus
embedding_dim = len(next(iter(embeddings_index.values())))

# Sukuriame matricas
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Svarbu: patikriname, ar vektoriaus ilgis sutampa su tikėtinu (kartais faile būna nelygumų)
        if len(embedding_vector) == embedding_dim:
            embedding_matrix[i] = embedding_vector
print(embedding_matrix)
# ==========================================
# 4. Modelio kūrimas su Keras 3
# ==========================================
model = keras.Sequential([
    # Naudojame Embedding sluoksnį su iš anksto apmokytais GloVe svoriais
    keras.layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        weights=[embedding_matrix], # Priskiriame mūsų paruoštą GloVe matricą
        trainable=False, # False reiškia, kad treniruotės metu šie vektoriai nebus keičiami (užšaldyti)
        mask_zero=True
    ), 
    keras.layers.LSTM(64, return_sequences=True), # Padidintas neuronų skaičius (8 buvo per mažai geram mokymuisi)
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dense(vocab_size) # Išvesties sluoksnis: tikimybės kiekvienam žodyno žodžiui
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

# ==========================================
# 5. Modelio treniravimas
# ==========================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=4)
]

# Treniruojame modelį paduodami PyTorch DataLoader tiesiai į Keras 3
model.fit(
    dataloader,
    epochs=35,
    callbacks=callbacks
)

# ==========================================
# 6. Teksto generavimo funkcija
# ==========================================
def generate_text(model, seed_text, gen_length=50, temperature=1.0):
    """
    Generuoja tekstą naudojant apmokytą modelį ir pradinę frazę (seed_text).
    """
    # Tokenizuojame pradinį tekstą lygiai taip pat, kaip treniravimo duomenis
    seed_words = re.findall(r'\b\w+\b|[^\w\s]', seed_text.lower())
    seq_ids = [word_index.get(w, 1) for w in seed_words]

    for _ in range(gen_length):
        # Užpildome/nukerpame seką iki reikiamo ilgio (seq_length), pridedant nulius priekyje ("pre")
        x = keras.utils.pad_sequences([seq_ids], maxlen=seq_length, padding="pre")
        
        # Gauname spėjimus paskutiniam sekos žodžiui (logits)
        logits = model.predict(x, verbose=0)[0, -1, :]
        
        # Pritaikome "temperatūrą". 
        # > 1.0 daro tekstą labiau atsitiktinį, < 1.0 - labiau nuspėjamą (konservatyvų).
        logits = np.array(logits) / temperature
        
        # Softmax funkcija rankiniu būdu, kad paverstume reikšmes į tikimybes nuo 0 iki 1
        exp_preds = np.exp(logits - np.max(logits)) # Atimame max išvengiant skaičiavimo klaidų (overflow)
        preds = exp_preds / np.sum(exp_preds)
        
        # Atsitiktinai (pagal apskaičiuotas tikimybes) pasirenkame sekančio žodžio indeksą
        next_id = np.random.choice(len(vocab), p=preds)
        seq_ids.append(int(next_id))

    # Konvertuojame sugeneruotų indeksų sąrašą atgal į žodžius
    return " ".join(inv_vocab.get(i, "") for i in seq_ids)

# ==========================================
# 7. Testavimas
# ==========================================
print("\n--- Sugeneruotas tekstas ---")
print(generate_text(model, "The war", gen_length=50, temperature=0.8))