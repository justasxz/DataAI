# # house regression competition
# import pandas as pd
# from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import cross_val_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# df = pd.read_csv('data/house_train.csv')

# # plot lotArea histogram to see distribution of lotArea
# # sns.histplot(df['LotArea'], bins=50, kde=True) # histplot - sukuria histogramą, x - nurodo x ašį, bins - nurodo kiek stulpelių naudoti, kde - nurodo ar pridėti kernel density estimation
# # plt.title('LotArea Distribution') # title - nustato grafiko pavadinimą
# # plt.xlabel('LotArea') # xlabel - nustato x ašies pavadinimą
# # plt.ylabel('Frequency') # ylabel - nustato y ašies pavadinimą
# # plt.grid() # grid - prideda tinklelio linijas
# # plt.show() # show - rodo grafiką

# # deal with LotArea outliers
# # df = df[df['LotArea'] < 100000] # filter - pasalina eilutes, kurios atitinka nurodytą sąlygą
# # three standard deviations from the mean
# # df = df[df['LotArea'] < df['LotArea'].mean() + 3 * df['LotArea'].std()] # filter - pasalina eilutes, kurios atitinka nurodytą sąlygą, mean - apskaiciuoja vidurki, std - apskaiciuoja standartini nuokrypi
# # df = df[df['LotArea'] > df['LotArea'].mean() - 3 * df['LotArea'].std()] # filter - pasalina eilutes, kurios atitinka nurodytą sąlygą, mean - apskaiciuoja vidurki, std - apskaiciuoja standartini nuokrypi
# # quantile method
# # df = df[df['LotArea'] < df['LotArea'].quantile(0.99)] # filter - pasalina eilutes, kurios atitinka nurodytą sąlygą, quantile - apskaiciuoja nurodyta kvantili
# # df = df[df['LotArea'] > df['LotArea'].quantile(0.01)] # filter - pasalina eilutes, kurios atitinka nurodytą sąlygą, quantile - apskaiciuoja nurodyta kvantili

# print(df.describe())
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler() # MinMaxScaler - normalizuoja duomenis, kad jie būtų tarp 0 ir 1
# # df['LotArea'] = scaler.fit_transform(df[['LotArea']]) # fit_transform - normalizuoja duomenis, kad jie būtų tarp 0 ir 1, galima nurodyti tik tam tikrus stulpelius
# # sns.scatterplot(x=df['LotArea'], y=df['SalePrice']) # scatterplot - sukuria sklaidos diagramą, x - nurodo x ašį, y - nurodo y ašį
# # plt.title('LotArea vs SalePrice') # title - nustato grafiko pavadinimą
# # plt.xlabel('LotArea') # xlabel - nustato x ašies pavadinimą
# # plt.ylabel('SalePrice') # ylabel - nustato y ašies pavadinimą
# # plt.grid() # grid - prideda tinklelio linijas
# # plt.show() # show - rodo grafiką
# # for column in df.select_dtypes(include=['int64', 'float64']).columns: # select_dtypes - pasirenka tik nurodytus duomenu tipus, columns - nurodo stulpelius
# #     sns.boxplot(x=df[column]) # boxplot - sukuria dėžės diagramą, x - nurodo x ašį
# #     plt.title(f'{column} Boxplot') # title - nustato grafiko pavadinimą
# #     plt.xlabel(column) # xlabel - nustato x ašies pavadinimą
# #     plt.grid() # grid - prideda tinklelio linijas
# #     plt.show() # show - rodo grafiką
# # sns.boxplot(x=df['LotArea']) # boxplot - sukuria dėžės diagramą, x - nurodo x ašį
# # plt.title('LotArea Boxplot') # title - nustato grafiko pavadinimą
# # plt.xlabel('LotArea') # xlabel - nustato x ašies pavadinimą
# # plt.grid() # grid - prideda tinklelio linijas
# # plt.show() # show - rodo grafiką

# # sns.histplot(df['LotArea'], bins=50, kde=True) # histplot - sukuria histogramą, x - nurodo x ašį, bins - nurodo kiek stulpelių naudoti, kde - nurodo ar pridėti kernel density estimation
# # plt.title('LotArea Distribution') # title - nustato grafiko pavadinimą
# # plt.xlabel('LotArea') # xlabel - nustato x ašies pavadinimą
# # plt.ylabel('Frequency') # ylabel - nustato y ašies pavadinimą
# # plt.grid() # grid - prideda tinklelio linijas
# # plt.show() # show - rodo grafiką
# # print(df)
# # df = pd.get_dummies(df)
# # # df = df.select_dtypes(include=['int64', 'float64']) # select_dtypes - pasirenka tik nurodytus duomenu tipus
# # df.info()
# # df.to_csv('data/house_train_encoded.csv', index=False) # to_csv - issaugo DataFrame i csv faila, index - nurodo ar issaugoti indexa, jei False - issaugo be indexo
# # # # print(df)
# # df.dropna(inplace=True) # dropna - pasalina eilutes su null reiksmemis, inplace - pakeicia originala, jei False - grazina nauja DataFrame, thresh - nurodo kiek ne null reiksmiu turi buti eiluteje, kad ji butu issaugota
# # # df.info()
# # df.set_index('Id', inplace=True) # set_index - nustato nurodyta stulpeli kaip index, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# # scaler = RobustScaler() # RobustScaler - normalizuoja duomenis, kad jie būtų tarp 0 ir 1, bet naudoja median ir interquartile range, kad butu atsparus outlieriams
# # X, y = df.drop('SalePrice', axis=1), df['SalePrice'] # drop - pasalina nurodyta stulpeli, axis - nurodo ar pasalinti eilutes (axis=0) ar stulpelius (axis=1), inplace - pakeicia originala, jei False - grazina nauja DataFrame
# # X = scaler.fit_transform(X) # fit_transform - normalizuoja duomenis, kad jie būtų tarp 0 ir 1, galima nurodyti tik tam tikrus stulpelius
# # print(X)

# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.metrics import mean_absolute_error

# # # model = KNeighborsRegressor(n_neighbors=5) # KNeighborsRegressor - sukuria K artimiausių kaimynų regresoriaus modelį, n_neighbors - nurodo kiek artimiausių kaimynų naudoti prognozavimui
# # # scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error') # cross_val_score - apskaiciuoja modelio tiksluma naudojant cross validation, cv - nurodo kiek daliu padalinti duomenis, scoring - nurodo kokia metrika naudoti tikslumo apskaiciavimui
# # # print(scores)
# # # cv_score = -scores.mean() # mean - apskaiciuoja vidurki, neg_mean_absolute_error - apskaiciuoja vidurki, bet grazina neigiama reiksme, todel reikia padaryti teigiama
# # # print(cv_score)
# # k_values = list(range(1, 31))
# # accuracies = []

# # for k in k_values:
# #     model = KNeighborsRegressor(n_neighbors=k) # KNeighborsRegressor - sukuria K artimiausių kaimynų regresoriaus modelį, n_neighbors - nurodo kiek artimiausių kaimynų naudoti prognozavimui
# #     scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error') # cross_val_score - apskaiciuoja modelio tiksluma naudojant cross validation, cv - nurodo kiek daliu padalinti duomenis, scoring - nurodo kokia metrika naudoti tikslumo apskaiciavimui
# #     cv_score = -scores.mean() # mean - apskaiciuoja vidurki, neg_mean_absolute_error - apskaiciuoja vidurki, bet grazina neigiama reiksme, todel reikia padaryti teigiama
# #     accuracies.append(cv_score)
# #     print(f'K={k}, CV Score: {cv_score}')

# # # from matplotlib import pyplot as plt
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(k_values, accuracies, marker='o') # plot - sukuria lin
# # # # ijinę diagramą, x - nurodo x ašį, y - nurodo y ašį, marker - nurodo kokį markerį naudoti taškams
# # # plt.title('KNN Regression CV Score for Different K Values')
# # # plt.xlabel('K Value')
# # # plt.ylabel('CV Score (Negative Mean Absolute Error)')
# # # plt.xticks(k_values)
# # # plt.grid()
# # # plt.show()
# # # get the best k value
# # best_k = k_values[accuracies.index(min(accuracies))] # index - apskaiciuoja nurodytos reiksmes indexa, min - apskaiciuoja maziausia reiksme
# # print(f'Best K Value: {best_k}, CV Score: {min(accuracies)}')

# # model = KNeighborsRegressor(n_neighbors=best_k) # KNeighborsRegressor - sukuria K artimiausių kaimynų regresoriaus modelį, n_neighbors - nurodo kiek artimiausių kaimynų naudoti prognozavimui
# # model.fit(X, y) # fit - train

# # # test data
# # df_test = pd.read_csv('data/house_test.csv')
# # df_test = pd.get_dummies(df_test)
# # df_test.set_index('Id', inplace=True) # Geriau indeksą nustatyti prieš sulygiuojant stulpelius

# # # Pridedame trūkstamus stulpelius, kurių nėra testavimo duomenyse
# # missing_cols = set(df.columns) - set(df_test.columns)
# # for col in missing_cols:
# #     if col != 'SalePrice': 
# #         df_test[col] = 0

# # # ATKOMENTUOTA EILUTĖ: Išrikiuojame testavimo duomenų stulpelius, kad jie idealiai atitiktų treniravimo duomenis
# # # Pastaba: 'SalePrice' stulpelio df_test neturi, todėl jį išmetame iš sąrašo, pagal kurį lygiuojame
# # df_test = df_test[df.columns.drop('SalePrice')] 

# # df_test.fillna(0, inplace=True) 

# # # Dabar transformacija ir prognozė veiks be klaidų!
# # df_test_scaled = scaler.transform(df_test) 
# # predictions = model.predict(df_test_scaled) 

# # submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': predictions})
# # submission.to_csv('house_submission.csv', index=False)


import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

# colored_image= cv.imread('cat.0.jpg', cv.IMREAD_COLOR_RGB) # imread - nuskaito paveikslėlį, cv.IMREAD_COLOR - nuskaito paveikslėlį spalvotą
# display the image
# plt.imshow(colored_image)
# plt.show()
colored_image = cv.imread('cat.0.jpg', cv.IMREAD_COLOR_RGB)
unclored = cv.cvtColor(colored_image, cv.COLOR_RGB2GRAY) # cvtColor - konvertuoja paveikslėlį į kitą spalvų erdvę, cv.COLOR_RGB2GRAY - konvertuoja iš RGB į pilką
plt.imshow(unclored, cmap='gray') # imshow - rodo paveiksl
# plt.imshow(uncolored_image)
plt.show()
# pixel_array = unclored # convert the image to a numpy array
# print(pixel_array.shape)