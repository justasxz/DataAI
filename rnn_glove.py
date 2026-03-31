import os
# CRITICAL: Backendą būtina nustatyti prieš importuojant keras!
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# --- 1. Nuskaitome tekstą ---
file_path = "data/war_peace.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

max_chars = 50000
raw_text = raw_text[:max_chars]

# --- 2. Tokenizacija ---
words = re.findall(r'\b\w+\b|[^\w\s]', raw_text.lower())
vocab = ["<pad>", "<unk>"] + list(dict.fromkeys(words))
word_index = {w: i for i, w in enumerate(vocab)}
inv_vocab = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

word_seq = [word_index.get(w, 1) for w in words]

# --- 3. GloVe Embeddings užkrovimas ---
def load_glove_embeddings(file_path, word_index, vocab_size):
    print("Kraunami GloVe vektoriai...")
    embeddings_index = {}
    embedding_dim = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            if embedding_dim == 0:
                embedding_dim = len(coefs)

    # Sukuriame matricą (vocab_size x embedding_dim)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    hits = 0
    misses = 0

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            # Žodžiai, kurių nėra GloVe, lieka nuliai (arba galima inicializuoti atsitiktinai)
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
            misses += 1
    
    print(f"Rasta žodžių: {hits}, Nerasta (OOV): {misses}")
    return embedding_matrix, embedding_dim

# Kelias iki tavo failo
glove_path = "data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
embedding_matrix, embed_dim = load_glove_embeddings(glove_path, word_index, vocab_size)

# --- 4. PyTorch Dataset ir DataLoader ---
seq_length = 50
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

# --- 5. Modelio kūrimas su Keras 3 ---
# Jei turime GloVe matricą, naudojame ją inicializacijai
embedding_layer = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embed_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix) if embedding_matrix is not None else "uniform",
    trainable=False, # Užšaldome, kad nepersitreniruotų iš pradžių (galima keisti į True)
    mask_zero=True
)

model = keras.Sequential([
    embedding_layer,
    keras.layers.LSTM(128, return_sequences=True), # Padidinau neuronų skaičių, nes 8 per mažai rimtam tekstui
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

# --- 6. Treniravimas ---
callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=4)]
model.fit(dataloader, epochs=20, callbacks=callbacks)

# --- 7. Teksto generavimas ---
def generate_text(model, seed_text, gen_length=50, temperature=1.0):
    seed_words = re.findall(r'\b\w+\b|[^\w\s]', seed_text.lower())
    seq_ids = [word_index.get(w, 1) for w in seed_words]

    for _ in range(gen_length):
        x = keras.utils.pad_sequences([seq_ids], maxlen=seq_length, padding="pre")
        logits = model.predict(x, verbose=0)[0, -1, :]
        
        logits = np.array(logits) / temperature
        exp_preds = np.exp(logits - np.max(logits))
        preds = exp_preds / np.sum(exp_preds)
        
        next_id = np.random.choice(len(vocab), p=preds)
        seq_ids.append(int(next_id))

    return " ".join(inv_vocab.get(i, "") for i in seq_ids)

print("\n--- Sugeneruotas tekstas ---")
print(generate_text(model, "The prince was", gen_length=30, temperature=0.7))