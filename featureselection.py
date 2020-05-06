import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv("fish_participant.csv")
print(data.shape)

data.drop(["Species"], axis=1)
X = data.iloc[:, 1:6]  # independent columns
y = data.iloc[
    :, 0
]  # target column i.e price range#apply SelectKBest class to extract top 10 best features
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
print(fit.scores_)

features = fit.transform(X)
print(features[0:5, :])
