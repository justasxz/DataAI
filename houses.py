# house regression competition
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/house_train.csv')
# print(df)

df = df.select_dtypes(include=['int64', 'float64']) # select_dtypes - pasirenka tik nurodytus duomenu tipus
# print(df)
df.dropna(inplace=True) # dropna - pasalina eilutes su null reiksmemis, inplace - pakeicia originala, jei False - grazina nauja DataFrame, thresh - nurodo kiek ne null reiksmiu turi buti eiluteje, kad ji butu issaugota
# df.info()
df.set_index('Id', inplace=True) # set_index - nustato nurodyta stulpeli kaip index, inplace - pakeicia originala, jei False - grazina nauja DataFrame
scaler = RobustScaler() # RobustScaler - normalizuoja duomenis, kad jie būtų tarp 0 ir 1, bet naudoja median ir interquartile range, kad butu atsparus outlieriams
X, y = df.drop('SalePrice', axis=1), df['SalePrice'] # drop - pasalina nurodyta stulpeli, axis - nurodo ar pasalinti eilutes (axis=0) ar stulpelius (axis=1), inplace - pakeicia originala, jei False - grazina nauja DataFrame
X = scaler.fit_transform(X) # fit_transform - normalizuoja duomenis, kad jie būtų tarp 0 ir 1, galima nurodyti tik tam tikrus stulpelius
print(X)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# model = KNeighborsRegressor(n_neighbors=5) # KNeighborsRegressor - sukuria K artimiausių kaimynų regresoriaus modelį, n_neighbors - nurodo kiek artimiausių kaimynų naudoti prognozavimui
# scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error') # cross_val_score - apskaiciuoja modelio tiksluma naudojant cross validation, cv - nurodo kiek daliu padalinti duomenis, scoring - nurodo kokia metrika naudoti tikslumo apskaiciavimui
# print(scores)
# cv_score = -scores.mean() # mean - apskaiciuoja vidurki, neg_mean_absolute_error - apskaiciuoja vidurki, bet grazina neigiama reiksme, todel reikia padaryti teigiama
# print(cv_score)
k_values = list(range(1, 31))
accuracies = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k) # KNeighborsRegressor - sukuria K artimiausių kaimynų regresoriaus modelį, n_neighbors - nurodo kiek artimiausių kaimynų naudoti prognozavimui
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error') # cross_val_score - apskaiciuoja modelio tiksluma naudojant cross validation, cv - nurodo kiek daliu padalinti duomenis, scoring - nurodo kokia metrika naudoti tikslumo apskaiciavimui
    cv_score = -scores.mean() # mean - apskaiciuoja vidurki, neg_mean_absolute_error - apskaiciuoja vidurki, bet grazina neigiama reiksme, todel reikia padaryti teigiama
    accuracies.append(cv_score)
    print(f'K={k}, CV Score: {cv_score}')

# from matplotlib import pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, accuracies, marker='o') # plot - sukuria lin
# # ijinę diagramą, x - nurodo x ašį, y - nurodo y ašį, marker - nurodo kokį markerį naudoti taškams
# plt.title('KNN Regression CV Score for Different K Values')
# plt.xlabel('K Value')
# plt.ylabel('CV Score (Negative Mean Absolute Error)')
# plt.xticks(k_values)
# plt.grid()
# plt.show()
# get the best k value
best_k = k_values[accuracies.index(min(accuracies))] # index - apskaiciuoja nurodytos reiksmes indexa, min - apskaiciuoja maziausia reiksme
print(f'Best K Value: {best_k}, CV Score: {min(accuracies)}')

model = KNeighborsRegressor(n_neighbors=best_k) # KNeighborsRegressor - sukuria K artimiausių kaimynų regresoriaus modelį, n_neighbors - nurodo kiek artimiausių kaimynų naudoti prognozavimui
model.fit(X, y) # fit - train

# test data
df_test = pd.read_csv('data/house_test.csv')
df_test = df_test.select_dtypes(include=['int64', 'float64']) # select_dtypes - pasirenka tik nurodytus duomenu tipus
df_test.fillna(0,inplace=True) # dropna - pasalina eilutes su null reik
df_test.set_index('Id', inplace=True) # set_index - nustato nurodyta stulpeli kaip index, inplace - pakeicia originala, jei False - grazina nauja DataFrame
df_test_scaled = scaler.transform(df_test) # transform - normalizuoja duomenis, kad jie būtų tarp 0 ir 1, galima nurodyti tik tam tikrus stulpelius
predictions = model.predict(df_test_scaled) # predict - prognozuoja reiksmes pagal
# print(predictions)

submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': predictions}) # sukuria DataFrame su dviem stulpeliais, Id - nurodo eiluciu numerius, SalePrice - nurodo prognozuotas reiksmes
submission.to_csv('house_submission.csv', index=False) # to_csv - issaugo Data