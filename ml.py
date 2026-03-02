from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/titanic_train.csv')
# df.info()
model = LogisticRegression(solver='liblinear') # LogisticRegression - sukuria logistinės regresijos modelį, C - reguliavimo parametras, max_iter - maksimalus iteracijų skaičius
# map male/female to 0/1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # TIK JEIGU YRA DU variantai
df = df.select_dtypes(include=['int64', 'float64']) # select_dtypes - pasirenka tik nurodytus duomenu tipus
df.set_index('PassengerId', inplace=True) # set_index - nustato nurodyta stulpeli kaip index, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# print(df.head())
# df.sort_values('Age', inplace=True) # sort_values - surikiuoja duomenis pagal nurodyta stulpeli, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# print(df.head())
# df.info()
# df.drop('Age', axis=1, inplace=True) # drop - pasalina nurodyta stulpeli, axis - nurodo ar pasalinti eilutes (axis=0) ar stulpelius (axis=1), inplace - pakeicia originala, jei False - grazina nauja DataFrame
# df['Age'] = df['Age'].fillna(df['Age'].mean()) # fillna - uzpildo null reiksmes nurodytu skaiciu, galima nurodyti skirtingas reiksmes skirtingiems stulpeliams
# most logical way to deal with age in titanic dataset
df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('mean')) # fillna - uzpildo null reiksmes nurodytu skaiciu, galima nurodyti skirtingas reiksmes skirtingiems stulpeliams
# print(df.head(20))

train_X, val_X, train_y, val_y = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.30, random_state=42) # train_test_split - padalina duomenis i mokymo ir testavimo dalis, test_size - nurodo kiek procentu duomenu skirti testavimui, random_state - nustato atsitiktinumo sėklą, kad rezultatai būtų atkuriami
# test_X, val_X, test_y, val_y = train_test_split(val_X, val_y, test_size=0.50, random_state=42) # train_test_split - padalina duomenis i mokymo ir testavimo dalis, test_size - nurodo kiek procentu duomenu skirti testavimui, random_state - nustato atsitiktinumo sėklą, kad rezultatai būtų atkuriami

# print(train_X.shape, train_y.shape)
# print(val_X.shape, val_y.shape)
# print(train_X)
# print(train_y)

    # X, y = df.drop('Survived', axis=1), df['Survived'] # drop - pasalina nurodyta stulpeli, axis - nurodo ar pasalinti eilutes (axis=0) ar stulpelius (axis=1), inplace - pakeicia originala, jei False - grazina nauja DataFrame
model.fit(train_X, train_y) # fit - train
# print(X)
# test_data = [[3,0,33,0,0,7.5]] # PassengerId=266, Sex=male (0), Pclass=2, Age=36, SibSp=0, Parch=0, Fare=10.5
# prediction = model.predict(test_data)
# print(prediction)

# test_data = [[1,1,11,2,2,45.5], [3,0,33,0,0,7.5]] # PassengerId=266, Sex=male (0), Pclass=2, Age=36, SibSp=0, Parch=0, Fare=10.5
# prediction = model.predict(test_data)
# print(prediction)

df_test = pd.read_csv('data/titanic_test.csv')
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1}) # TIK JEIGU YRA DU variantai
df_test = df_test.select_dtypes(include=['int64', 'float64']) # select_dtypes - pasirenka tik nurodytus duomenu tipus
# df_test.set_index('PassengerId', inplace=True) # set_index - nustato nurodyta stulpeli kaip index, inplace - pakeicia originala, jei False - grazina nauja DataFrame
# we should not use test data to fillna, because we should not use test data to train our model, but we can use train data to fillna in test data
# if we use df to fill it will not fill because index mismatch, so we need to use groupby and transform to fillna in test data
df_test['Age'] = df_test['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('mean')) # fillna - uzpildo null reiksmes nurodytu skaiciu, galima nurodyti skirtingas reiksmes skirtingiems stulpeliams
df_test.info()
df_test.set_index('PassengerId', inplace=True) # set_index - nustato nurodyta stulpeli kaip index, inplace - pakeicia originala, jei False - grazina nauja DataFrame
print(df_test.head(20))
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean()) # fillna - uzpildo null reiksmes nurodytu skaiciu, galima nurodyti skirtingas reiksmes skirtingiems stulpeliams

pred_val = model.predict(val_X) # predict - prognozuoja reiksmes pagal mokymo duomenis
print(pred_val)
from sklearn.metrics import accuracy_score
print(accuracy_score(val_y, pred_val)) # accuracy_score - apskaiciuoja tiksluma, lygina tikslas reiksmes su prognozuotomis reiksmes, grazina tiksluma procentais


# predictions = model.predict(df_test)
# # print(predictions)

# result_df = pd.DataFrame({'PassengerId': df_test.index, 'Survived': predictions}) # sukuria DataFrame su nurodytais stulpeliais
# print(result_df)
# result_df.to_csv('titanic_predictions.csv', index=False) # to_csv - issaugo DataFrame i csv faila, index - nurodo ar issaugoti indexa, jei False - neissaugoti indexa
# print(model.coef_)
# test = [[3, 1, 22.0, 1, 0, 7.25]] answers [3, 1]

