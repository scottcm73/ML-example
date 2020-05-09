import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR

### None of the models are great... But model3 is the best, and it has its own function.
def dummify(df):
    train_df_d = pd.get_dummies(df["Species"])
    print(train_df_d)
    df = df.drop(columns=["Species"])

    df = df.merge(train_df_d, left_index=True, right_index=True)
    return df


def do_prediction(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred

# Redundant with sklearn

def get_and_prep_data(csv_path):
    this_df = pd.read_csv(csv_path)
    print(this_df.describe())
    df=dummify(this_df)
    return df


# Winning Model -- Because model3 is not strictly a sklearn model I needed to make it have its own function.
def model3(csv_path):
    df=get_and_prep_data(csv_path)
    model3 = df.assign(
        y_pred=(df["Length1"] + df["Length2"] + df["Length3"])
        / 3
        * df["Height"]
        * df["Width"]
        * 0.26
    )
    print(model3)
    y_prediction = model3[["y_pred"]].to_numpy()
    y_pred = y_prediction.ravel()
    return y_pred


csv_path="fish_participant.csv"
df = get_and_prep_data(csv_path)
# Multiple Linear Regression (not the best...)
print(df.columns)
X_train = df.iloc[:, df.columns != "Weight"]
y_train = df.iloc[:, 0]
model = LinearRegression()
model.fit(X_train, y_train)
coeff_df = pd.DataFrame(model.coef_, X_train.columns, columns=["Coefficient"])
print(coeff_df)


csv_path="fish_holdout_demo.csv"
df2 = get_and_prep_data(csv_path)
# Make X_test with Dummy variables
X_test = df2.iloc[:, df2.columns != "Weight"]
#First column is Actual Weight
y_test = df2.iloc[:, 0]
y_true = y_test
y_pred = do_prediction(X_test, model)

# Ensure that values above zero.
y_pred = abs(y_pred)
print(y_pred)

mse = mean_squared_error(y_true, y_pred)
print(mse)

# SVR (not good)
n_samples, n_features = 10, 5
rng = np.random.RandomState(5)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
model2 = SVR(C=1.0, epsilon=0.1)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print("SVR -- bad model")
print(y_pred)
mse = mean_squared_error(y_true, y_pred)
print(mse)

#  Length Average * Height * Width *.27 

print("Model of average Length * Width * Height * .26 ")

# Numpy Array of Predicted Weights
# Call model3 function
y_pred = model3(csv_path)
print("Weight Predictions: ")
print(y_pred)

mse3 = mean_squared_error(y_true, y_pred)

# Still quite high??
# print(f"Mean Squared Error: {mse3}")


# Random Forest


csv_path="fish_participant.csv"
df = get_and_prep_data(csv_path)
csv_path2="fish_holdout_demo.csv"
df2 = get_and_prep_data(csv_path2)
sc=StandardScaler()

X_train = df.iloc[:, df.columns != "Weight"]
y_train = df.iloc[:, 0]
X_test = df2.iloc[:, df.columns != "Weight"]
y_test = df2.iloc[:, 0]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(y_pred)
mse = mean_squared_error(y_true, y_pred)
print(mse)