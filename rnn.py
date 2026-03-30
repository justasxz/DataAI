import os
# CRITICAL: Backendą būtina nustatyti prieš importuojant keras!
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# 1. Nuskaitome tekstą iš failo
file_path = "data/war_peace.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Apribojame tekstą, kad sutaupytume laiko/atminties (kaip tavo originale)
max_chars = 50000
raw_text = raw_text[:max_chars] # pirmas 50 tūkstančių raidžių iš teksto

# 2. Tokenizacija be TensorFlow
# Naudojame paprastą RegEx, kad atskirtume žodžius ir skyrybos ženklus.
words = re.findall(r'\b\w+\b|[^\w\s]', raw_text.lower()) # pavercia i mazasas ir pasalina 
vocab = ["<pad>", "<unk>"] + list(dict.fromkeys(words)) # lieka Unikalūs žodžiai ir tada padarom, kad pad būtų užpildas, o unk nežinomas žodis
word_index = {w: i for i, w in enumerate(vocab)} # [labas, kaip, sekasi] -> {"labas": 1, 'kaip': 2}
inv_vocab = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Konvertuojame tekstą į indeksų seką
word_seq = [word_index.get(w, 1) for w in words] # 1 yra <unk> indeksas
print(word_seq)
# 3. PyTorch Dataset ir DataLoader (pakeičia tf.data.Dataset)
seq_length = 50 # kiek zodziu imsime, kad spetumem sekanti zodi (rnn atsimena pries tai ejususius zodzius)
batch_size = 64 # kiek tokiu seku modelis apdoros vienu metu

# Labas, man labai patinka kaledos, per jas gaunu daug dovanu ir buna daug sviesu. Taip pat, kaledos yra laikas susitikti su seima

class TextDataset(Dataset):
    def __init__(self, sequence, seq_len):
        self.sequence = sequence
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequence) - self.seq_len # jis seka kie kdar zodziu liko
    def __getitem__(self, idx):
        # Input (X): nuo idx iki idx + seq_len
        # Target (Y): pasislinkęs per vieną tokeną
        x = self.sequence[idx : idx + self.seq_len] # [0:50 zodziai, 1:51, 2:52]
        y = self.sequence[idx + 1 : idx + self.seq_len + 1] # [51,52,53]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = TextDataset(word_seq, seq_length) # zodziu skaicius, kiek zodziu bus naudojama vienam spejimui
# drop_last=True užtikrina, kad batch_size visada būtų vienodas
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 4. Modelio kūrimas su Keras 3
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True), # [5,9,24,17,4...] -> [(7,2),(1,4),(8,15)]
    keras.layers.LSTM(8, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(8, return_sequences=True),
    keras.layers.Dense(vocab_size) # 10000 [0.001,0.0000015,0.02]
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

# 5. EarlyStopping ir treniravimas
callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=4)
]

epochs = 20

# Keras 3 model.fit natively priima PyTorch DataLoader!
model.fit(
    dataloader,
    epochs=epochs,
    callbacks=callbacks
)

# 6. Teksto generavimo funkcija
def generate_text(model, seed_text, gen_length=50, temperature=1.0):
    """
    Generuoja tekstą naudojant apmokytą modelį ir NumPy (vietoj tf.random).
    """
    # Paruošiame pradinį tekstą
    seed_words = re.findall(r'\b\w+\b|[^\w\s]', seed_text.lower())
    seq_ids = [word_index.get(w, 1) for w in seed_words]

    for _ in range(gen_length):
        # Keras 3 utils turi pad_sequences (pakeičia senąjį tf.keras.preprocessing)
        x = keras.utils.pad_sequences([seq_ids], maxlen=seq_length, padding="pre")
        
        # Nuspėjame sekantį žodį. predict() grąžina NumPy masyvą (net ir su Torch backendu)
        logits = model.predict(x, verbose=0)[0, -1, :]
        
        # Pritaikome temperatūrą ir Softmax rankiniu būdu (su NumPy)
        logits = np.array(logits) / temperature
        exp_preds = np.exp(logits - np.max(logits)) # Atimame max dėl stabilumo
        preds = exp_preds / np.sum(exp_preds)
        
        # Parenkame sekantį indeksą remdamiesi tikimybėmis
        next_id = np.random.choice(len(vocab), p=preds)
        seq_ids.append(int(next_id))

    # Atstatome indeksus atgal į žodžius
    return " ".join(inv_vocab.get(i, "") for i in seq_ids)

# 7. Išbandome
print("\n--- Sugeneruotas tekstas ---")
print(generate_text(model, "My name is", gen_length=50, temperature=1.0))