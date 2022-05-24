import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pip
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

df = pd.read_csv('titanic.csv')

# df_name=df.columns
# df = df.drop(['Earnings'], axis=1)

df.info()

# df = df.replace('-', np.NaN)

missing_values = df.isnull().sum()
print(missing_values)

sns.heatmap(df.isna(), yticklabels=False, cbar=False)

plt.figure(figsize=(10, 6))
sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)

# df = df.replace(',','', regex=True)

# df = df.replace(r'[%]+$', '', regex=True)

df.head()

df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
df.head()

# print(df[['Pclass', 'Cabin']])
# print(len(set(df['Cabin'])))

missing_values = df.isnull().sum()
missingvp = 100 * (missing_values) / 891

print(missingvp)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df[['Age']])
df[['Age']] = imputer.transform(df[['Age']])

df = df.dropna(subset=['Embarked'], axis=0)
df.info()

X = df.iloc[:, 1:].values

print(X)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
print(X[0])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [ 1, 6, 7])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct1.fit_transform(df))
print(X[1])

ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct2.fit_transform(df))
print(X[1])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(df)

df_corr = df[['from Open', 'Perf Week', 'Gap', 'SMA20', 'SMA200', 'RSI',
              'Volatility W', 'Perf Quart', '52W High', 'Perf Half', 'Rel Volume',
              'Perf YTD', 'Volatility M', 'Perf Year', 'Oper M', 'SMA50', 'Volume',
              'Profit M', 'ROA']].corr(method='pearson')
print(df_corr)

corr_matrix = df.corr().abs()
print(corr_matrix)

sorted_mat = corr_matrix.unstack().sort_values()

print(sorted_mat)

df_corr = df.corr(method='pearson')
print(df_corr)

mask = np.triu(np.ones_like(df_corr, dtype=bool))

f, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(df_corr, mask=mask, cmap='jet', vmax=.3, center=0, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .8});
plt.title('Correlation analysis ');

sorted_mat.to_excel('corr20.xlsx')

df = df.replace(np.NaN, 0)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)

feature_names = [f"feature {i}" for i in range(X.shape[1])]
importances = regressor.feature_importances_
forest_importances = pd.Series(importances, index=feature_names)
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

df.head()

from sklearn.inspection import permutation_importance

result = permutation_importance(
    regressor, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
