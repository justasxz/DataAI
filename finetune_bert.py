# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# import torch

# # 1. Paruošiame modelį ir tokenizatorių (2 klasėms: pvz. Teigiamas/Neigiamas)
# model_name = "google-bert/bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # 2. Minimalūs duomenys (tik demonstracijai)
# texts = ["I love this product!", "This is the worst thing ever.", "Amazing experience.", "I hate it."]
# labels = [1, 0, 1, 0] # 1 - teigiamas, 0 - neigiamas

# # Tokenizuojame duomenis
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# # Sukuriame paprastą duomenų rinkinio formatą
# class SimpleDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# dataset = SimpleDataset(inputs, labels)

# # 3. Fine-tuning nustatymai
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     logging_steps=1,
# )

# # 4. Paleidžiame apmokymą
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
# )

# print("Pradedamas fine-tuning...")
# trainer.train()
# print("Apmokymas baigtas! Modelis dabar geriau supranta jūsų specifinius duomenis.")