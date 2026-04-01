from transformers import pipeline

# Naudojame "text-generation" užduotį ir GPT tipo modelį
generator = pipeline("text-generation", model="gpt2")

print("Sveiki! Aš esu generatyvinis modelis. Parašykite pradžią, o aš pratęsiu.")

while True:
    user_input = input("\nJūsų tekstas (arba 'q' išeiti): ")
    if user_input.lower() == 'q':
        break
    
    # Generuojame tekstą
    # max_length - kiek iš viso žodžių/tokenų bus rezultate
    # num_return_sequences - kiek variantų sukurti
    results = generator(user_input, max_length=50, num_return_sequences=1, truncation=True)

    print("\nGeneruotas tekstas:")
    print(results[0]['generated_text'])