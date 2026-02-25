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
# unsorted_series_with_nan = pd.Series([5, 2, 3, np.nan, 1, 4, np.nan])
# sorted_series = unsorted_series_with_nan.sort_values() # sort_values - surusiuoja
# print(sorted_series)

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', np.nan, "Jonas"],
    'Age': [25, 30, 35, 40, 45, 50,np.nan],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio'],
    'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 300000],
    'Department': [np.nan, "HR", np.nan, np.nan, "IT", np.nan, np.nan]
})

# print(df)
# print(df.shape)
# df.info()
# print(df.describe()) # describe - pateikia statistines reiksmes apie skaitinius stulpelius, count - kiek yra ne null reiksmiu, mean - vidurkis, std - standartinis nuokrypis, min - minimumas, 25% - pirmasis kvartilis, 50% - antrasis kvartilis (mediana), 75% - treciasis kvartilis, max - maksimumas
# print(df['Salary'].quantile(0.25,interpolation='nearest')) # quantile - pateikia nurodyta kvartili, interpolation - nurodo kaip apskaiciuoti kvartili, lower - naudoja apacia reiksme, higher - naudoja virsutine reiksme, midpoint - naudoja vidurki tarp apacios ir virsutines reiksmes
# print(df.describe(include='all')) # include - nurodo kokio tipo stulpelius apibrezti, object - tekstiniai stulpeliai, all - visi stulpeliai
# print(df['Name'])

# print(df.describe()['Age'])

# print(df[['Name', 'City']]) # pasiekiama kelis stulpelius pagal pavadinima
# filter
# print(df[df['Age'] > 30]) # isfiltruojami eilutes, kuriose amzius didesnis uz 30
# rename
# df.rename(columns={'Name': 'Full Name', 'Age': 'Age in Years'}, inplace=True) # rename - pervadina stulpelius, columns - nurodo kuriuos stulpelius pervadinti, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# print(df)
# print(df.isnull())
# df['Age'] = df['Age'].fillna(100) # fillna - uzpildo null reiksmes nurodytu skaiciu
# print(df)
# df.fillna({'Name': 'Unknown', 'Age': 100}, inplace=True) # fillna - uzpildo null reiksmes nurodytu skaiciu, galima nurodyti skirtingas reiksmes skirtingiems stulpeliams
# print(df)
# df.dropna(inplace=True,thresh=3) # dropna - pasalina eilutes su null reiksmemis, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# print(df)
# df.dropna(axis=1, thresh=4, inplace=True) # dropna - pasalina stulpelius su null reiksmemis, axis - nurodo ar pasalinti eilutes (axis=0) ar stulpelius (axis=1), inplace - pakeicia originala, jei False - grazina nauja DataFrame
# print(df)
# df['Country'] = 'USA' # prideda nauja stulpeli su nurodyta reiksme
# df['Country'] = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'Lietuva'] # prideda nauja stulpeli su nurodyta reiksme, galima nurodyti skirtingas reiksmes skirtingiems eilutems
# print(df)
# df2 = df.copy() # copy - sukuria nauja DataFrame, kuris yra kopija originalo, pakeitimai viename neitakos kitam
# df2.rename(columns={'Name': 'Full Name', 'Age': 'Age in Years'}, inplace=True) # rename - pervadina stulpelius, columns - nurodo kuriuos stulpelius pervadinti, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# df = pd.concat([df, df2], ignore_index=True) # concat - sujungia du DataFrame i viena, ignore_index - ignoruoja originalius indexus ir sukuria naujus nuo 0 iki n-1
# print(df)
df = pd.read_csv(r'data/titanic_train.csv') # read_csv - nuskaito duomenis is csv failo ir sukuria DataFrame
print(df)
# df.info()
# pd.read_sql_query

print(df.groupby('Sex')['PassengerId'].count()) # groupby - grupuoja duomenis pagal nurodyta stulpeli, count - suskaiciuoja kiek yra ne null reiksmiu kiekvienoje grupeje
print(df.groupby('Pclass')['PassengerId'].count()) # groupby - grupuoja duomenis pagal nurodyta stulpeli, count - suskaiciuoja kiek yra ne null reiksmiu kiekvienoje grupeje
print(df.groupby('Pclass')['Age'].mean()) # groupby - grupuoja duomen