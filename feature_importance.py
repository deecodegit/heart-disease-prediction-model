import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from train_models import preprocessor, X, models, x_test, y_test, x_test_proc, nn

# ----------------------------------------------------------
# helper — save plots
# ----------------------------------------------------------
def save_plot(vals, names, title, outpath):
    sorted_idx = np.argsort(vals)[::-1]
    vals_sorted = vals[sorted_idx]
    names_sorted = np.array(names)[sorted_idx]

    plt.figure(figsize=(10,6))
    plt.bar(names_sorted, vals_sorted)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ----------------------------------------------------------
# get feature names after preprocessing
# ----------------------------------------------------------
def get_feature_names():
    # numeric
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()

    # categorical (one-hot)
    cat_transformer = preprocessor.named_transformers_['cat']['onehot']
    cat_features = cat_transformer.get_feature_names_out()

    # combine
    all_features = np.concatenate([num_features, cat_features])
    return all_features

feature_names = get_feature_names()

# ----------------------------------------------------------
# 1. logistic regression importance
# ----------------------------------------------------------
log_reg = models["logistic"]
log_coeffs = log_reg.named_steps["clf"].coef_[0]

save_plot(
    log_coeffs,
    feature_names,
    "logistic regression - feature importance (coefficients)",
    "static/plots/logistic_feature_importance.png"
)

print("logistic regression importance saved.")


# ----------------------------------------------------------
# 2. decision tree importance
# ----------------------------------------------------------
dt = models["decision_tree"].named_steps["clf"]
dt_importances = dt.feature_importances_

save_plot(
    dt_importances,
    feature_names,
    "decision tree - feature importance",
    "static/plots/decision_tree_feature_importance.png"
)

print("decision tree importance saved.")


# ----------------------------------------------------------
# 3. random forest importance
# ----------------------------------------------------------
rf = models["random_forest"].named_steps["clf"]
rf_importances = rf.feature_importances_

save_plot(
    rf_importances,
    feature_names,
    "random forest - feature importance",
    "static/plots/random_forest_feature_importance.png"
)

print("random forest importance saved.")


# ----------------------------------------------------------
# 4. neural network — permutation importance
# ----------------------------------------------------------
print("running permutation importance for neural network (this may take ~20 sec)...")

def nn_predict(x):
    return nn.predict(x, verbose=0).flatten()

perm_result = permutation_importance(
    estimator=None,
    X=x_test_proc,
    y=y_test,
    scoring="f1",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
    # we manually define prediction via nn
)

# but sklearn permutation_importance needs an estimator
# workaround: use its code manually

nn_scores = []
for i in range(x_test_proc.shape[1]):
    x_permuted = x_test_proc.copy()
    np.random.shuffle(x_permuted[:, i])
    pred = (nn_predict(x_permuted) > 0.5).astype(int)
    f1 = (2 * (pred * y_test).sum()) / (pred.sum() + y_test.sum() + 1e-9)
    nn_scores.append(f1)

nn_importances = np.max(nn_scores) - np.array(nn_scores)

save_plot(
    nn_importances,
    feature_names,
    "neural network - permutation importance",
    "static/plots/nn_feature_importance.png"
)

print("neural network importance saved.")
print("all feature importance plots generated successfully.")