from sklearn.neighbors import KNeighborsClassifier

# KNeighborsRegressor()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset('titanic')
print(df)
# convert to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = df.select_dtypes(include=['int64', 'float64']) # select_dtypes - pasirenka tik nurodytus duomenu tipus
# print(df)

x_train, x_val, y_train, y_val = train_test_split(df.drop('survived', axis=1), df['survived'], test_size=0.2, random_state=42) # train_test_split - padalina duomenis i mokymo ir testavimo duomenis, test_size - nurodo kiek procentu duomenu naudoti testavimui, random_state - nurodo atsitiktinumo sėklą, kad rezultatai būtų atkuriami

model = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=3) # DecisionTreeClassifier - sukuria sprendimų medžio klasifikatoriaus modelį, max_depth - nurodo maksimalų medžio gylį
# model_knn = KNeighborsClassifier(n_neighbors=5)
# model_knn.fit(x_train, y_train) # fit - train
# pred_val_knn = model_knn.predict(x_val) # predict - prognozuoja reiksmes pagal mokymo duomenis
# print(accuracy_score(y_val, pred_val_knn)) # accuracy_score - apskaici
model.fit(x_train, y_train) # fit - train
pred_val = model.predict(x_val) # predict - prognozuoja reiksmes pagal mokymo duomenis
print(accuracy_score(y_val, pred_val)) # accuracy_score - apskaiciuoja tiksluma, lygina tikslas reiksmes su prognozuotomis reiksmes, grazina tiksluma procentais
tree.plot_tree(model) # plot_tree - sukuria sprendimų medžio diagramą, model - nurodo modelį, kuris bus naudojamas diagramai kurti
plt.show() # show - rodo grafiką