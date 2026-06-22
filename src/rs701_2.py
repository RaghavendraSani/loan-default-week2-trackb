import numpy as np
import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print("Train shape: ")
print(train.shape)

print("\nTest shape: ")
print(test.shape)

print("\ncolumns: ")
print(train.columns.tolist())

print("\nData Types: ")
print(train.dtypes)

print("\nMissing Values: ")
print(train.isnull().sum())

print("\ntarget distribution: ")
print(train["default"].value_counts())
print(train["default"].value_counts(normalize=False))

categorical_cols = [
    "education",
    "home_status",
    "loan_purpose",
    "region",
    "noise_cat_1"
]
for col in categorical_cols:
    print("\n"+"="*50)
    print(col)
    print(train[col].value_counts(dropna=False))

print("\n"+"="*50)
print("num feature summ. ")
print(train.describe())

print("\nTarget distribution:")
print(train["default"].value_counts())

print("\nTarget percentages:")
print(train["default"].value_counts(normalize=True))

print("\nUnique values per categorical column")

for col in categorical_cols:
    print(f"\n{col}")
    print(train[col].unique())

def handle_missing_values(train_df, test_df):
    numerical_cols = [
        "annual_income",
        "credit_score",
        "employment_years",
        "num_late_payments",
        "savings_balance"
    ]

    for col in numerical_cols:
        median_value = train_df[col].median()

        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    train_df["education"] = train_df["education"].fillna("Missing")
    test_df["education"] = test_df["education"].fillna("Missing")
    return train_df, test_df

def drop_id_column(train_df, test_df):
    train_df = train_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])
    return train_df, test_df

train, test = handle_missing_values(train, test)
train, test = drop_id_column(train, test)

print("\nmissing val after cleaning: ")
print("train isnull: ")
print(train.isnull().sum())
print("\ntest isnull: ")
print(test.isnull().sum())

def encode_categorical_features(train_df, test_df):
    categorical_cols = [
        "education",
        "home_status",
        "loan_purpose",
        "region",
        "noise_cat_1"
    ]
    combined = pd.concat(
        [train_df.drop(columns=["default"]), test_df], axis=0
    )
    combined = pd.get_dummies(
        combined,
        columns=categorical_cols,
        dtype=int
    )

    train_rows = len(train_df)

    X_train = combined.iloc[:train_rows]
    X_test = combined.iloc[train_rows:]
    X_train["default"] = train_df["default"].values

    return X_train, X_test

train, test = encode_categorical_features(train, test)

print("\nTrain shape after encoding:")
print(train.shape)

print("\nTest shape after encoding:")
print(test.shape)

print("\nColumns after encoding:")
print(train.columns.tolist())
