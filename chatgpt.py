# from transformers import pipeline

# # Naudojame "text-generation" užduotį ir GPT tipo modelį
# generator = pipeline("text-generation", model="gpt2")

# print("Sveiki! Aš esu generatyvinis modelis. Parašykite pradžią, o aš pratęsiu.")

# while True:
#     user_input = input("\nJūsų tekstas (arba 'q' išeiti): ")
#     if user_input.lower() == 'q':
#         break
    
#     # Generuojame tekstą
#     # max_length - kiek iš viso žodžių/tokenų bus rezultate
#     # num_return_sequences - kiek variantų sukurti
#     results = generator(user_input, max_length=50, num_return_sequences=1, truncation=True)

#     print("\nGeneruotas tekstas:")
#     print(results[0]['generated_text'])


from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# 1. Užkrauname modelį ir tokenizatorių
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 neturi specialaus "padding" tokeno (priešingai nei BERT), 
# todėl nustatome jį naudoti sakinio pabaigos (eos) simbolį.
tokenizer.pad_token = tokenizer.eos_token

# 2. Paruošiame duomenis (Pavyzdžiui: mokome modelį specifinių faktų)
data = [
    "The secret ingredient of the grand pizza is extra blue cheese.",
    "Our company, TechFlow, was founded in 2024 in Vilnius.",
    "To reset the device, hold the power button for ten seconds.",
    "The capital of Mars is the Great Red Crater."
]

# Tokenizavimo funkcija
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

# Sukuriame Dataset objektą
dataset = Dataset.from_dict({"text": data})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Data Collator - jis pasirūpina, kad modelis mokytųsi spėti kitą žodį (labels = inputs)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Apmokymo nustatymai
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=10,        # Mažam duomenų kiekiui reikia daugiau epochų
    per_device_train_batch_size=2,
    save_steps=10,
    logging_steps=1,
)

# 5. Trainer paleidimas
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

print("Pradedamas GPT-2 fine-tuning...")
trainer.train()

# 6. Testuojame apmokytą modelį
print("\n--- Testas: Ar modelis išmoko naują informaciją? ---")
prompt = "The secret ingredient of the grand pizza is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=20, num_return_sequences=1)

print("Sugeneruotas atsakymas:", tokenizer.decode(outputs[0], skip_special_tokens=True))