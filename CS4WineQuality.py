import pandas as pd
df = pd.read_csv("D:/intern/internship/Wine Quality/winequalityN.csv")
# print(df.head())
# print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

df['fixed acidity'].fillna((df['fixed acidity'].mean()), inplace=True)
df['volatile acidity'].fillna((df['volatile acidity'].mean()), inplace=True)
df['citric acid'].fillna((df['citric acid'].mean()), inplace=True)
df['residual sugar'].fillna((df['residual sugar'].mean()), inplace=True)
df['chlorides'].fillna((df['chlorides'].mean()), inplace=True)
df['pH'].fillna((df['pH'].mean()), inplace=True)
df['sulphates'].fillna((df['sulphates'].mean()), inplace=True)

# print(df.isnull().sum())
# df.hist(bins=20, figsize=(10, 10))
# plt.show()

rf = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=1)

le = LabelEncoder()
le.fit(df['type'])
df['type'] = le.transform(df['type'])
print(df)

df = df.drop('density', axis=1)
print(df.isnull().sum())

#Defining X and Y
X = df.drop('quality', axis=1)
Y = df['quality']

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(13).plot(kind='barh')
# plt.show()

from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)
# print(Counter(Y))

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['volatile acidity'])
plt.show()

import seaborn as sns
sns.boxplot(df['quality'])
plt.show()
out = ['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'pH', 'sulphates', 'alcohol' ]
for i in out:
    print(df[i])
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    print(upper)
    print(lower)
    out1 = df[df[i] < lower].values
    out2 = df[df[i] > upper].values
    df[i].replace(out1, lower, inplace=True)
    df[i].replace(out2, upper, inplace=True)
    sns.boxplot(df[i])
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

func = [rf, dtc]
for item in func:
    item.fit(X_train, y_train)
    y_pred = item.predict(X_test)
    print(item)
    print(accuracy_score(y_test, y_pred))


