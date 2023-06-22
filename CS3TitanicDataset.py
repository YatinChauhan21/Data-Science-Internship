import pandas as pd
df = pd.read_csv("C:/Users/yatin/Downloads/archive/Titanic-Dataset.csv")

from sklearn.preprocessing import LabelEncoder
# from sklearn.ensembele import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.naive_bays import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# print(df.head)
x = df.drop('Embarked',axis=1)
x = x.drop('Cabin',axis=1)
x = x.drop('PassengerId',axis=1)
x = x.drop('Ticket',axis=1)
x = x.drop('Name',axis=1)
# print(x)
# print(df.isnull().sum())
x['Age'].fillna((x['Age'].mean()),inplace=True)
x['Fare'].fillna((x['Fare'].mean()),inplace=True)
# print(x.isnull().sum())

le = LabelEncoder()
x['Sex']=le.fit_transform(df['Sex'])
# print(df['Sex'])

x=x.drop('Survived', axis=1)
y= df['Survived']
from collections import  Counter
from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
x,y = sms.fit_resample(x,y)
# print(Counter(y))
#
from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['Fare'])
plt.show()

print(x['Age'])
Q1 = x['Age'].quantile(0.25)
Q3 = x['Age'].quantile(0.75)

IQR = Q3-Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)
out1 =x[x['Age']<lower].values
out2 = x[x['Age']>upper].values
x['Age'].replace(out1,lower,inplace=True)
x['Age'].replace(out2,upper,inplace=True)

print(x['Age'])

print(x['Fare'])
Q1 = x['Fare'].quantile(0.25)
Q3 = x['Fare'].quantile(0.75)

IQR = Q3-Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)
out1 =x[x['Fare']<lower].values
out2 = x[x['Fare']>upper].values
x['Fare'].replace(out1,lower,inplace=True)
x['Fare'].replace(out2,upper,inplace=True)

print(x['Fare'])

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr = LogisticRegression()
pca = PCA(n_components=2)


pca.fit(x)
X = pca.transform(x)
print(X)
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.3)
logr.fit(X_train,y_train)
y_pred= logr.predict(X_test)
a=(accuracy_score(y_test,y_pred))
print(accuracy_score)

print(a*100)

