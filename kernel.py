# Imports
import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Figures inline and set visualization style
sns.set()

# Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# View head
data.info()
data.Name.tail()
# Extract Title from Name, store in column and plot barplot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
data.tail()

# Did they have a Cabin?
data['Has_Cabin'] = ~data.Cabin.isnull()

# View head of data
data.head()
# Drop columns and view head
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()
data.info()
# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.mean())
data['Fare'] = data.Fare.fillna(data.Fare.mean())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()
data.describe()
# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=7, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=9, labels=False)

data.head()
data = data.drop(['Age', 'Fare'], axis=1)
data.head()
data['Fam_Size'] = data.Parch + data.SibSp
# Drop columns
data = data.drop(['SibSp','Parch'], axis=1)
data.head()
# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()
# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values


x_test = data_test.values

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu',input_shape=(12, ))
)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='Nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, epochs=120, batch_size=24)

predictions = np.round(model.predict(test))
predictions = pd.DataFrame(predictions)
result = pd.concat([df_train[["PassengerId"]], predictions], axis = 1)
predictions.to_csv("result.csv", index=False)
