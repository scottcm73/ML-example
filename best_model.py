import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def dummify(df):
    train_df_d = pd.get_dummies(df["Species"])
    df = df.drop(columns=["Species"])
    df = df.merge(train_df_d, left_index=True, right_index=True)
    return df

# Redundant with sklearn

def get_and_prep_data(csv_path):
    this_df = pd.read_csv(csv_path)
    df=dummify(this_df)
    return df

def do_rf_prediction(test_csv_path):
    df2 = get_and_prep_data(test_csv_path)
    X_test = df2.iloc[:, df.columns != "Weight"]
    y_test = df2.iloc[:, 0]
    X_test = sc.transform(X_test)
    y_pred = regressor.predict(X_test)
    return y_pred, y_test

csv_path="fish_participant.csv"
df = get_and_prep_data(csv_path)
test_csv_path = "fish_holdout_demo.csv"

sc=StandardScaler()

X_train = df.iloc[:, df.columns != "Weight"]
y_train = df.iloc[:, 0]
X_train = sc.fit_transform(X_train)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)

## Get the random forest prediction and the y_test result.
y_pred, y_test = do_rf_prediction(test_csv_path)
print(y_pred)

