import os
# CRITICAL: Backendą būtina nustatyti prieš importuojant keras!
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# ==========================================
# 1. DUOMENŲ NUSKAITYMAS IR PARUOŠIMAS
# ==========================================
file_path = "data/war_peace.txt"

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Apribojame tekstą, kad sutaupytume laiko/atminties
max_chars = 50000
raw_text = raw_text[:max_chars]

# ==========================================
# 2. TOKENIZACIJA
# ==========================================
# Naudojame paprastą RegEx, kad atskirtume žodžius ir skyrybos ženklus.
words = re.findall(r'\b\w+\b|[^\w\s]', raw_text.lower())
vocab = ["<pad>", "<unk>"] + list(dict.fromkeys(words))
word_index = {w: i for i, w in enumerate(vocab)}
inv_vocab = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Konvertuojame tekstą į indeksų seką
word_seq = [word_index.get(w, 1) for w in words]

# ==========================================
# 3. PYTORCH DATASET IR DATALOADER
# ==========================================
seq_length = 50 # Maksimalus žodžių skaičius, kurį modelis matys vienu metu
batch_size = 64

class TextDataset(Dataset):
    def __init__(self, sequence, seq_len):
        self.sequence = sequence
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequence) - self.seq_len

    def __getitem__(self, idx):
        x = self.sequence[idx : idx + self.seq_len]
        y = self.sequence[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = TextDataset(word_seq, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ==========================================
# 4. MODELIO KŪRIMAS (TRANSFORMER)
# ==========================================

# Kuriame savo Keras sluoksnį, kuris sujungia žodžių ir pozicijų vektorius
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        # x forma: (batch_size, seq_len)
        seq_len = keras.ops.shape(x)[1]
        
        # Sukuriame pozicijas [0, 1, ..., seq_len-1]
        positions = keras.ops.arange(start=0, stop=seq_len, step=1)
        
        # Ištraukiame vektorius. Keras automatiškai pritaiko "broadcasting",
        # kad sudėtų (batch, seq, dim) su (seq, dim)
        return self.token_emb(x) + self.pos_emb(positions)

# Įvestis priima dinaminį ilgį (None)
inputs = keras.Input(shape=(None,), dtype="int32")

# Naudojame mūsų naująjį sluoksnį!
x = TokenAndPositionEmbedding.call(maxlen=seq_length, vocab_size=vocab_size, embed_dim=128)(inputs)

# Transformerio Blokas
attention_output = keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(x, x, use_causal_mask=True)

x = keras.layers.LayerNormalization()(x + attention_output)

ffn_output = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.LayerNormalization()(x + ffn_output)

# Išvesties sluoksnis
outputs = keras.layers.Dense(vocab_size)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

print("\n--- Modelio Architektūra ---")
model.summary()

# ==========================================
# 5. TRENIRAVIMAS
# ==========================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=4)
]

epochs = 20

print("\n--- Pradedamas treniravimas ---")
model.fit(
    dataloader,
    epochs=epochs,
    callbacks=callbacks
)

# ==========================================
# 6. TEKSTO GENERAVIMAS BE PADDING'O
# ==========================================
def generate_text(model, seed_text, gen_length=50, temperature=1.0):
    seed_words = re.findall(r'\b\w+\b|[^\w\s]', seed_text.lower())
    seq_ids = [word_index.get(w, 1) for w in seed_words]

    for _ in range(gen_length):
        # Apkerpame seką, kad ji neviršytų maksimalaus ilgio (seq_length)
        x_input = seq_ids[-seq_length:] 
        
        # Sukuriame (1, kintamas_ilgis) formos NumPy masyvą
        x = np.array([x_input]) 
        
        # Nuspėjame sekantį žodį
        logits = model.predict(x, verbose=0)[0, -1, :]
        
        # Pritaikome temperatūrą ir Softmax
        logits = np.array(logits) / temperature
        exp_preds = np.exp(logits - np.max(logits))
        preds = exp_preds / np.sum(exp_preds)
        
        # Pasirenkame sekantį žodį
        next_id = np.random.choice(len(vocab), p=preds)
        seq_ids.append(int(next_id))

    return " ".join(inv_vocab.get(i, "") for i in seq_ids)

# ==========================================
# 7. REZULTATAI
# ==========================================
print("\n--- Sugeneruotas tekstas ---")
print(generate_text(model, "My name is", gen_length=50, temperature=0.8))