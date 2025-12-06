# train_models.py
# beginner-friendly training script for heart disease project
# - loads raw csv from your windows path
# - preprocesses (impute, one-hot, scale)
# - trains logistic regression, decision tree, random forest (with quick randomized search)
# - trains a simple keras neural network
# - evaluates, saves metrics + plots, and saves models
#
# usage:
#   python train_models.py
#
# note: this script assumes you installed required libs globally (no venv).
# change DATA_PATH to your csv if it's in another place.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------
# config - change only this if needed
# ---------------------------
DATA_PATH = r"C:\Users\Devanshi Sahu\data analytics\projects\heart-disease-ML-project\data\heart.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
PLOTS_DIR = os.path.join("static", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------
# helper functions
# ---------------------------
def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def plot_and_save_roc(y_true, prob_dict, outpath):
    plt.figure(figsize=(8,6))
    for name, probs in prob_dict.items():
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (auc={roc_auc:.3f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(outpath)
    plt.close()

def plot_and_save_confusion(y_true, y_pred, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------------------------
# 1. load data
# ---------------------------
print("loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("first rows:")
print(df.head())
print("shape:", df.shape)
print("missing values per column:")
print(df.isnull().sum())

# ---------------------------
# 2. basic cleaning
# ---------------------------
df = df.drop_duplicates()
print("after dropping duplicates shape:", df.shape)

# ensure target exists
if "target" not in df.columns:
    raise ValueError("no 'target' column found in dataset")

# ---------------------------
# 3. features / target
# ---------------------------
X = df.drop(columns=["target"])
y = df["target"]

# detect categorical-like columns automatically: numeric cols with few unique values
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_candidates = []
for c in numeric_cols.copy():
    if X[c].nunique() <= 6:
        cat_candidates.append(c)
        numeric_cols.remove(c)

# also include object dtype columns (if any)
obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
categorical_cols = sorted(list(set(cat_candidates + obj_cols)))

print("numeric columns:", numeric_cols)
print("categorical columns:", categorical_cols)

# ---------------------------
# 4. preprocessing pipeline
# ---------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

# ---------------------------
# 5. train/test split
# ---------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("train shape:", x_train.shape, "test shape:", x_test.shape)

# ---------------------------
# 6. models to train (pipelines)
# ---------------------------
models = {}

models["logistic"] = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
])

models["decision_tree"] = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# random forest we will tune quickly with randomized search
rf_pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])
models["random_forest"] = rf_pipeline

# ---------------------------
# 7. quick randomized search for random forest (fast)
# ---------------------------
from scipy.stats import randint as sp_randint

rf_param_dist = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10]
}

print("running quick randomized search for random forest (n_iter=6, cv=3)...")
rf_search = RandomizedSearchCV(rf_pipeline, param_distributions=rf_param_dist,
                               n_iter=6, cv=3, scoring="f1", n_jobs=-1, random_state=42)
rf_search.fit(x_train, y_train)
print("best rf params:", rf_search.best_params_)
models["random_forest"] = rf_search.best_estimator_

# ---------------------------
# 8. train logistic + decision tree (simple fit)
# ---------------------------
print("training logistic regression and decision tree (simple fit)...")
models["logistic"].fit(x_train, y_train)
models["decision_tree"].fit(x_train, y_train)

# ---------------------------
# 9. train a simple keras neural network
# ---------------------------
# build a preprocessing-only pipeline to transform x_train into numeric matrix for keras
print("preprocessing x_train for keras...")
x_train_proc = preprocessor.fit_transform(x_train)
x_test_proc = preprocessor.transform(x_test)

# build a tiny neural net (simple, quick)
def build_simple_nn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

print("building and training keras neural net (10 epochs, small)...")
nn = build_simple_nn(x_train_proc.shape[1])
# early stopping to prevent long runs
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = nn.fit(x_train_proc, y_train, validation_split=0.1, epochs=10, batch_size=16, callbacks=[es], verbose=1)
# save keras model
nn_path = os.path.join(MODEL_DIR, "neural_net.h5")
nn.save(nn_path)
print("saved keras model to", nn_path)

# ---------------------------
# 10. evaluate all models
# ---------------------------
results = {}
probs = {}

# evaluate logistic
for name, pipe in models.items():
    print(f"evaluating {name} ...")
    # sklearn pipelines accept raw df input (they apply preprocessor inside)
    y_pred = pipe.predict(x_test)
    try:
        y_prob = pipe.predict_proba(x_test)[:,1]
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }
    probs[name] = y_prob
    print(name, "metrics:", results[name])

# keras nn evaluate
print("evaluating keras neural net on test set...")
nn_pred_prob = nn.predict(x_test_proc).ravel()
nn_pred = (nn_pred_prob >= 0.5).astype(int)
results["neural_net"] = {
    "accuracy": float(accuracy_score(y_test, nn_pred)),
    "precision": float(precision_score(y_test, nn_pred, zero_division=0)),
    "recall": float(recall_score(y_test, nn_pred, zero_division=0)),
    "f1": float(f1_score(y_test, nn_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, nn_pred_prob))
}
probs["neural_net"] = nn_pred_prob
print("neural_net metrics:", results["neural_net"])

# ---------------------------
# 11. save sklearn models and scaler / pipeline
# ---------------------------
print("saving sklearn models...")
for name, pipe in models.items():
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(pipe, path)
    print("saved", name, "to", path)

# also save preprocessor so we can use it for keras inputs later if needed
preproc_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preproc_path)
print("saved preprocessor to", preproc_path)

# ---------------------------
# 12. save metrics & comparisons
# ---------------------------
metrics_path = os.path.join(REPORT_DIR, "metrics.json")
save_json(results, metrics_path)
print("saved metrics json to", metrics_path)

# also save a simple csv comparison
comp_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"model"})
comp_path = os.path.join(REPORT_DIR, "model_comparisons.csv")
comp_df.to_csv(comp_path, index=False)
print("saved model comparison csv to", comp_path)

# ---------------------------
# 13. plot roc curves and confusion matrices
# ---------------------------
roc_path = os.path.join(PLOTS_DIR, "roc_curve_all.png")
plot_and_save_roc(y_test, probs, roc_path)
print("saved roc plot to", roc_path)

# confusion matrix for the best model by f1
best_model_name = max(results.items(), key=lambda x: x[1]["f1"])[0]
print("best model by f1:", best_model_name)
best_pred = None
if best_model_name == "neural_net":
    best_pred = (probs["neural_net"] >= 0.5).astype(int)
else:
    best = models[best_model_name]
    best_pred = best.predict(x_test)

cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{best_model_name}.png")
plot_and_save_confusion(y_test, best_pred, cm_path)
print("saved confusion matrix to", cm_path)

# ---------------------------
# 14. save feature importance if available (random forest or tree)
# ---------------------------
def save_feature_importance(pipe, outpath, top_n=15):
    # attempt to extract feature names and importances
    try:
        # get feature names after preprocessing
        # preprocessor is ColumnTransformer; we build names manually
        num_names = numeric_cols
        # get onehot names
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        onehot_names = list(onehot.get_feature_names_out(categorical_cols))
        feature_names = np.array(list(num_names) + onehot_names)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            idx = np.argsort(importances)[::-1][:top_n]
            import pandas as pd
            fi_df = pd.DataFrame({
                "feature": feature_names[idx],
                "importance": importances[idx]
            })
            fi_df.to_csv(os.path.join(REPORT_DIR, "feature_importances.csv"), index=False)
            # plot
            plt.figure(figsize=(8,6))
            plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
            plt.title("feature importance (top {})".format(top_n))
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()
            print("saved feature importance to", outpath)
    except Exception as e:
        print("could not save feature importances:", e)

fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
# try for random forest
if "random_forest" in models:
    save_feature_importance(models["random_forest"], fi_path)

# ---------------------------
# done
# ---------------------------
print("training script finished. results saved in", REPORT_DIR, "and plots in", PLOTS_DIR)
