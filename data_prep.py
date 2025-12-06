# data_prep.py
# -------------------------------------
# this script loads the heart.csv dataset,
# cleans it, encodes categories, scales numbers,
# splits into train/test sets, and saves everything.
# -------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# -------------------------------------
# 1. load the dataset
# -------------------------------------

print("loading the data...")

# load the csv from your path
# load the csv from your local path
df = pd.read_csv(
    r"C:\Users\Devanshi Sahu\data analytics\projects\heart-disease-ML-project\data\heart.csv"
)

print("\nfirst 5 rows:")
print(df.head())

print("\ndata shape (rows, columns):")
print(df.shape)

print("\nmissing values in each column:")
print(df.isnull().sum())

# -------------------------------------
# 2. basic cleaning
# -------------------------------------

# (this dataset usually has no missing values,
# but let's clean anyway just in case)

df = df.drop_duplicates()

print("\nafter dropping duplicates:", df.shape)

# -------------------------------------
# 3. separate features and target
# -------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------------
# 4. identify numeric + categorical columns
# -------------------------------------

# numeric features (columns with numbers)
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# categorical features — dataset uses these
categorical_features = ["cp", "thal", "slope"]  # manually known categorical columns

print("\nnumeric columns:", numeric_features)
print("categorical columns:", categorical_features)

# -------------------------------------
# 5. preprocessing: scaling + encoding
# -------------------------------------

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -------------------------------------
# 6. split into train/test
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\ntrain set size:", X_train.shape)
print("test set size:", X_test.shape)

# -------------------------------------
# 7. fit preprocessing and transform data
# -------------------------------------

print("\napplying scaling + encoding ...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("processed train shape:", X_train_processed.shape)
print("processed test shape:", X_test_processed.shape)

# -------------------------------------
# 8. create data folder if not exists
# -------------------------------------

if not os.path.exists("data"):
    os.makedirs("data")

# save processed datasets
pd.DataFrame(X_train_processed).to_csv("data/X_train_processed.csv", index=False)
pd.DataFrame(X_test_processed).to_csv("data/X_test_processed.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nall processed files saved in /data folder!")
print("data prep completed successfully ✔️")
