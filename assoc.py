# Įdiekite biblioteką, jei jos dar neturite:
# pip install efficient-apriori

# Įdiekite biblioteką, jei jos dar neturite:
# pip install efficient-apriori

from efficient_apriori import apriori

# 1. Duomenų paruošimas
transactions = [
    ['duona', 'sviestas', 'sūris'],
    ['duona', 'miltai'],
    ['sviestas', 'pienas', 'kava'],
    ['duona', 'sviestas', 'pienas'],
    ['sūris', 'pienas'],
    ['duona', 'sūris', 'sviestas'],
]

# 2. Dažnų derinių suradimas
min_support = 0.3 # 
min_confidence = 0.7 # 70%

itemsets, rules = apriori(
    transactions,
    min_support=min_support,
    min_confidence=min_confidence
)

# 3. Rezultatų atvaizdavimas
print("Dažni elementų deriniai (itemsets):")
for length, itemset in itemsets.items():
    print(f"\n{length}-elementų deriniai:")
    for items, support in itemset.items():
        print(f"  Elementai: {items}, support: {support:.2f}")

print("\nAsociacijų taisyklės:")
for rule in rules:
    print(rule)

def rekomenduoti(pasirinkti_itemai, rules):
    rekomendacijos = set()
    for rule in rules:
        # Paverčiame į set, nes lhs ir rhs yra tuple
        if set(rule.lhs).issubset(pasirinkti_itemai):
            rekomendacijos.update(set(rule.rhs))
    return rekomendacijos

# Naudotojo įvestis
print("\n--- Rekomendacijų sistema ---")
ivestis = input("Įveskite turimas prekes (atskirti kableliu, pvz. 'duona, sviestas'): ")
pasirinkti = set(item.strip() for item in ivestis.split(",") if item.strip())

# Gauti rekomendacijas
rekomendacijos = rekomenduoti(pasirinkti, rules)

if rekomendacijos:
    print(f"Pagal jūsų pasirinkimą {pasirinkti}, rekomenduojama dar įsigyti: {', '.join(rekomendacijos)}")
else:
    print("Šį kartą neturime rekomendacijų pagal jūsų pasirinkimą.")