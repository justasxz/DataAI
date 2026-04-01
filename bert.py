from transformers import pipeline

# Užkrauname "fill-mask" procesą su BERT
mask_filler = pipeline("fill-mask", model="google-bert/bert-base-uncased")

print("Sveiki! Įveskite sakinį anglų kalba su žodžiu [MASK].")
print("Pavyzdys: I want to [MASK] a new car.")

while True:
    user_input = input("\nJūsų sakinys (arba 'q' išeiti): ")
    if user_input.lower() == 'q':
        break
    
    if "[MASK]" not in user_input:
        print("Klaida: Sakinyje turi būti [MASK] žymė.")
        continue

    # Gauname rezultatus
    results = mask_filler(user_input)

    print("\nBERT siūlo:")
    for res in results:
        print(f"-> {res['sequence']} (tikimybė: {round(res['score'], 3)})")