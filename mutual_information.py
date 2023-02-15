import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv("autos.csv")
df.head()

X = df.copy()
y = X.pop("price")

X.drop(['stroke', 'bore', 'length', 'width', 'height', 'wheel_base'], axis=1, inplace = True)

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features='auto')
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores)  # show a few features with their MI scores
