import numpy as np
import pandas as pd
np.random.seed(42)

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

#print("Train shape: ")
#print(train.shape)

#print("\nTest shape: ")
#print(test.shape)

#print("\ncolumns: ")
#print(train.columns.tolist())

#print("\nData Types: ")
#print(train.dtypes)

#print("\nMissing Values: ")
#print(train.isnull().sum())

#print("\ntarget distribution: ")
#print(train["default"].value_counts())
#print(train["default"].value_counts(normalize=False))

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

#print("\n"+"="*50)
#print("num feature summ. ")
#print(train.describe())

#print("\nTarget distribution:")
#print(train["default"].value_counts())

#print("\nTarget percentages:")
#print(train["default"].value_counts(normalize=True))

#print("\nUnique values per categorical column")

#for col in categorical_cols:
    #print(f"\n{col}")
    #print(train[col].unique())

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

#print("\nmissing val after cleaning: ")
#print("train isnull: ")
#print(train.isnull().sum())
#print("\ntest isnull: ")
#print(test.isnull().sum())

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
        drop_first=True,
        dtype=int
    )

    train_rows = len(train_df)

    X_train = combined.iloc[:train_rows]
    X_test = combined.iloc[train_rows:]
    X_train = X_train.copy()
    X_train["default"] = train_df["default"].values

    return X_train, X_test

train, test = encode_categorical_features(train, test)

#print("\nTrain shape after encoding:")
#print(train.shape)

#print("\nTest shape after encoding:")
#print(test.shape)

#print("\nColumns after encoding:")
#print(train.columns.tolist())

def scale_features(train_df, test_df):

    feature_cols = [col for col in train_df.columns if col != "default"]

    for col in feature_cols:

        mean = train_df[col].mean()
        std = train_df[col].std()

        if std != 0:
            train_df[col] = (train_df[col] - mean) / std
            test_df[col] = (test_df[col] - mean) / std

    return train_df, test_df

train, test = scale_features(train, test)
#print("\nTrain head: ")
#print(train.head())

#print("\nTest head: ")
#print(test.head())

#print("\ntrain shape: ")
#print(train.shape)

#print("\ntest shape: ")
#print(test.shape)

X = train.drop(columns=["default"]).values
y = train["default"].values
X_test = test.values

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)

indices = np.arange(len(X))
np.random.shuffle(indices)

split_index = int(len(X) * 0.8)

train_idx = indices[:split_index]
val_idx = indices[split_index:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_features):
    weights = np.zeros(n_features)
    bias = 0.0
    return weights, bias

def forward_pass(X, weights, bias):
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    return predictions

weights, bias = initialize_parameters(X_train.shape[1])

predictions = forward_pass(
    X_train,
    weights,
    bias
)

print("\nFirst 10 predictions:")
print(predictions[:10])

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    loss = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
    return loss

loss = compute_loss(y_train, predictions)
print("\ninitial loss: ")
print(loss)