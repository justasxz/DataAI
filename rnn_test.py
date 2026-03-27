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
words = re.findall(r'\b\w+\b|[^\w\s]', raw_text.lower())
vocab = ["<pad>", "<unk>"] + list(dict.fromkeys(words)) # Unikalūs žodžiai
print(vocab)
word_index = {w: i for i, w in enumerate(vocab)}
inv_vocab = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Konvertuojame tekstą į indeksų seką
word_seq = [word_index.get(w, 1) for w in words] # 1 yra <unk> indeksas
# print(len(word_seq))
# print(len(words))

# Labas, kaip sekasi, kaip tavo diena ?