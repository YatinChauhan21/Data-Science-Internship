import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train = pd.read_json("D:/intern/internship/whats-cooking/train.json/train.json")
test = pd.read_json("D:/intern/internship/whats-cooking/test.json/test.json")
# print((train.head()))
# print(train.isnull().sum())
# print(test.head())
# print(test.isnull().sum())
# print(train['cuisine'].unique())


import matplotlib.pyplot as plt

plt.style.use('ggplot')
train['cuisine'].value_counts().plot(kind='bar')
# plt.show()

from collections import Counter

counters = {}
for cuisine in train['cuisine'].unique():
    counters[cuisine] = Counter()
    indices = (train['cuisine'] == cuisine)
    for ingredients in train[indices]['ingredients']:
        counters[cuisine].update(ingredients)


top10 = pd.DataFrame([[items[0] for items in counters[cuisine].most_common(10)] for cuisine in counters],
                     index=[cuisine for cuisine in counters],
                     columns=['top{}'.format(i) for i in range(1, 11)])
# print(top10)
train['all_ingredients'] = train['ingredients'].map(";".join)
print(train.head())

print(train['all_ingredients'].str.contains('garlic cloves'))


indices = train['all_ingredients'].str.contains('garlic cloves')
train[indices]['cuisine'].value_counts().plot(kind='bar',
                                                 title='garlic cloves as found per cuisine')
# plt.show()
# relative_freq = (train[indices]['cuisine'].value_counts() / train['cuisine'].value_counts())
# relative_freq.sort(inplace=True)
# relative_freq.plot(kind='bar')

import numpy as np
unique = np.unique(top10.values.ravel())
# print(unique)

fig, axes = plt.subplots(8, 8, figsize=(20, 20))
for ingredient, ax_index in zip(unique, range(64)):
    indices = train['all_ingredients'].str.contains(ingredient)
    relative_freq = (train[indices]['cuisine'].value_counts() /train['cuisine'].value_counts())
    relative_freq.plot(kind='bar', ax=axes.ravel()[ax_index], fontsize=7, title=ingredient)
# plt.show()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(train['all_ingredients'].values)
print(X.shape)
# print(list(cv.vocabulary_.keys())[:100])


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(train.cuisine)
# print(y[:100])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

print(logistic.score(X_test, y_test))


#https://flothesof.github.io/kaggle-whats-cooking-machine-learning.html