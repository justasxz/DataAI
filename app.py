# import numpy as np

# sarasas = [1,2,3,4,5,6,7,8,9,10,11,12]
# pakeistas = np.array(sarasas)
# pakeistas_matrica = pakeistas.reshape(3,4)
# # print(pakeistas)
# # print(pakeistas.shape)
# # print(pakeistas.dtype)

# # slicintas = pakeistas[1:3, 0:2] # eilutes, antras stulpelis (antras skaicius NE IMTINAI) (SKAICIUOJAM NUO NULIO)
# # print(slicintas)

# filtruotas = pakeistas[pakeistas > 5] # isfiltruojami skaiciai didesni uz 5
# print(filtruotas)

# pakeistas.

# print(np.sin(sarasas))
# print(np.sin(pakeistas))
# print(np.log(2.71))

# sarasas_a = sarasas + 2
# print(sarasas_a)
# pakeistas_a = pakeistas + 2
# print(pakeistas_a)
# pakeistas_matrica_a = pakeistas_matrica + 2
# print(pakeistas_matrica_a)

# pakeistas_b = pakeistas + np.array([7,2,3,4,5,6,7,8,9,10,11,12])
# print(pakeistas_b)

# # np.concatenate((pakeistas, pakeistas_a)) # sujungia du masyvus i viena
# print(np.concatenate((pakeistas, pakeistas_a)))

# def sudeti(a=1,b=2):
#     return a + b

import pandas as pd
import numpy as np

# serija = pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
# print(serija)
# print(serija.head())
# print(serija.size)
# print(serija.shape)
# print(serija.values)
# print(serija.index)
# print(serija['c']) # pasiekiama reiksme pagal indexa
# print(serija[0:2])
# print(serija['a':'c'])
# loc vs iloc
# loc - pasiekiama pagal indexa
# iloc - pasiekiama pagal pozicija
# print(serija.loc['a':'c'])
# print(serija.iloc[2]) # pasiekti pagal integer pozicija

# filtruota = serija[(serija > 3) | (serija < 2)] # isfiltruojami skaiciai didesni uz 3 arba mazesni uz 1
# print(filtruota) # | - arba & - ir 
# serija2 = serija * 2
# print(serija2)
# print(serija.mul(2))

# print(np.std(serija)) # standartinis nuokrypis, ddof - degrees of freedom, n-1
# print(serija.std())

# data_with_nan = pd.Series([1,2,3,np.nan,5,6,np.nan,8,9,10,11,12,13,14,15,np.nan])
# print(data_with_nan)
# print(data_with_nan.isnull())
# print(sum(data_with_nan.isnull())) # kiek yra null reiksmiu
# print(data_with_nan.size)
# grazintas = data_with_nan.dropna()
# print(grazintas)

# data_with_nan.dropna(inplace=True) # inplace - pakeicia originala, jei False - grazina nauja serija
# print(data_with_nan)

# def saraso_pakeitimas(sarasas):
#     nauja_kopija = sarasas.copy()
#     nauja_kopija.append(100)
#     return nauja_kopija

# saraso_pakeitimas(data_with_nan)

# def saraso_pakeitimas_inplace(sarasas):
#     sarasas.append(100)
#     return sarasas

# data_with_nan.fillna(100, inplace=True) # fillna - uzpildo null reiksmes nurodytu skaiciu
# print(data_with_nan)
# string_data_names_with_nan = pd.Series(['Alice', 'Bob', np.nan, 'Charlie', 'David', np.nan, 'Eve', "Bob", "Alice", "Bob"])
# # print(string_data_names_with_nan)
# # print(string_data_names_with_nan.str.upper())
# # pd.date_range(start='2024-01-01', end='2024-01-10')
# # print(pd.date_range(start='2024-01-01', end='2024-03-10'))

# print(string_data_names_with_nan.value_counts()) # suskaiciuoja kiek kartu pasikartoja kiekviena reiksme
# print(string_data_names_with_nan.unique())

# data_with_nan = pd.Series([1,2,3,np.nan,5,6,np.nan,8,9,10,11,12,13,14,15,np.nan])
# # data_with_nan = data_with_nan.apply(lambda x: x*2+3 if pd.notnull(x) else x) # apply - taiko funkcija kiekvienam elementui, lambda - anonimines funkcijos kurimas, pd.notnull - tikrina ar reiksme nera null
# # print(data_with_nan)
# data_with_nan = data_with_nan.map({np.nan: 0, 1: 10, 2: 20, 3: 30, 5: 50, 6: 60, 8: 80, 9: 90, 10: 100, 11: 110, 12: 120, 13: 130, 14: 140, 15: 150}) # map - pakeicia reiksmes pagal nurodyta zodyna, np.nan: 0 - pakeicia null reiksmes i 0
# print(data_with_nan)
unsorted_series_with_nan = pd.Series([5, 2, 3, np.nan, 1, 4, np.nan])
sorted_series = unsorted_series_with_nan.sort_values() # sort_values - surusiuoja
print(sorted_series)