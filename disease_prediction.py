"""
Disease Prediction from Medical Data
====================================
Predicts the probability that a patient has a disease from structured data
(age, symptoms, blood-test results, ...).

Datasets
  breast_cancer : UCI Breast Cancer Wisconsin (Diagnostic) - built in, no download
  heart         : UCI Heart Disease (Cleveland)  -> pass the CSV with --csv
  diabetes      : Pima Indians Diabetes          -> pass the CSV with --csv
  custom        : any CSV with a binary target   -> --csv file.csv --target column

Pipeline
  1. Load + clean (missing-value markers, binary target where 1 = disease)
  2. Preprocess inside the model pipeline (impute -> scale / one-hot) so that no
     information leaks from the test set or from cross-validation folds
  3. Models: Logistic Regression, SVM (RBF), Random Forest, XGBoost
     (scikit-learn gradient boosting is used automatically if xgboost is not installed)
  4. Hyper-parameter tuning with stratified 5-fold CV (ROC-AUC) on the training set only
  5. Evaluation on a held-out test set: accuracy, precision, recall (sensitivity),
     specificity, F1, ROC-AUC, PR-AUC, confusion matrix
  6. Screening threshold: lowest-cost cut-off that reaches the target sensitivity
     (default 95 %), chosen on training data only - missing a sick patient is worse
     than a false alarm
  7. Permutation feature importance for the best model

Usage
  python disease_prediction.py --dataset breast_cancer
  python disease_prediction.py --dataset heart    --csv heart.csv
  python disease_prediction.py --dataset diabetes --csv diabetes.csv
  python disease_prediction.py --dataset custom   --csv data.csv --target label
  python disease_prediction.py --predict new_patients.csv --dataset breast_cancer

Requirements: numpy pandas scikit-learn matplotlib joblib   (optional: xgboost)

DISCLAIMER: educational project. Not a medical device; never use for real diagnosis.
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

warnings.filterwarnings("ignore")
SEED = 42
TARGET_SENSITIVITY = 0.95


# ----------------------------------------------------------------------------
# 1. DATA LOADING  (positive class 1 = disease present)
# ----------------------------------------------------------------------------
def load_breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    X = d.data.copy()
    y = (d.target == 0).astype(int)              # sklearn: 0 = malignant -> flip so 1 = malignant
    return X, y, "malignant tumour"


def _read_csv(path):
    if not path:
        raise SystemExit("This dataset needs --csv path/to/file.csv")
    return pd.read_csv(path, na_values=["?", "NA", "N/A", "", " "])


def load_heart_data(path):
    """Cleveland / Kaggle heart-disease CSVs: target is 'target', 'num', 'condition', 'HeartDisease' or 'output'."""
    df = _read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = next((c for c in ["target", "num", "condition", "HeartDisease", "output", "goal"] if c in df.columns), None)
    if tcol is None:
        raise SystemExit(f"Could not find the target column in {list(df.columns)}; use --dataset custom --target <name>")
    y = (pd.to_numeric(df[tcol], errors="coerce").fillna(0) > 0).astype(int)   # UCI 'num' 1-4 -> disease
    return df.drop(columns=[tcol]), y, "heart disease"


def load_diabetes_data(path):
    """Pima Indians Diabetes CSV with an 'Outcome' column."""
    df = _read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = next((c for c in ["Outcome", "outcome", "class", "Class", "target"] if c in df.columns), df.columns[-1])
    y = df[tcol].astype(int)
    X = df.drop(columns=[tcol]).copy()
    # In this dataset a value of 0 means "not measured" for these columns
    for c in X.columns:
        if c.lower() in {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}:
            X[c] = X[c].replace(0, np.nan)
    return X, y, "diabetes"


def load_custom_data(path, target):
    df = _read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"--target {target!r} not in columns {list(df.columns)}")
    y = df[target]
    if y.dtype == object:
        y = (y.astype(str).str.lower().isin({"1", "yes", "y", "true", "positive", "disease", "malignant", "m"})).astype(int)
    else:
        y = (y > 0).astype(int)
    return df.drop(columns=[target]), y, "disease"


def load_data(args):
    if args.dataset == "breast_cancer":
        return load_breast_cancer_data()
    if args.dataset == "heart":
        return load_heart_data(args.csv)
    if args.dataset == "diabetes":
        return load_diabetes_data(args.csv)
    return load_custom_data(args.csv, args.target)


# ----------------------------------------------------------------------------
# 2. PREPROCESSING + MODELS
# ----------------------------------------------------------------------------
def make_preprocessor(X):
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("category", "string", "str", "bool")]
    # integer columns with very few distinct values (e.g. chest-pain type 1-4) are categories too
    cat += [c for c in X.columns if c not in cat and pd.api.types.is_integer_dtype(X[c]) and X[c].nunique() <= 4
            and X[c].nunique() > 2]
    num = [c for c in X.columns if c not in cat]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], remainder="drop")


def build_models(X, y_train):
    """name -> (pipeline, hyper-parameter search space)."""
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pre = lambda: make_preprocessor(X)
    models = {
        "Logistic Regression": (
            Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
        "SVM (RBF)": (
            Pipeline([("pre", pre()), ("clf", SVC(probability=True, class_weight="balanced", random_state=SEED))]),
            {"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]}),
        "Random Forest": (
            Pipeline([("pre", pre()), ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                                     n_jobs=-1, random_state=SEED))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 4, 8, 12],
             "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", 0.5]}),
    }
    if HAVE_XGB:
        models["XGBoost"] = (
            Pipeline([("pre", pre()), ("clf", XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                                           n_jobs=2, random_state=SEED))]),
            {"clf__n_estimators": [150, 300, 500], "clf__max_depth": [2, 3, 4, 6],
             "clf__learning_rate": [0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
             "clf__colsample_bytree": [0.7, 1.0]})
    else:
        models["Gradient Boosting (XGBoost stand-in)"] = (
            Pipeline([("pre", pre()), ("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                                             random_state=SEED))]),
            {"clf__max_iter": [100, 200, 400], "clf__max_depth": [2, 3, 4, None],
             "clf__learning_rate": [0.02, 0.05, 0.1], "clf__l2_regularization": [0, 1, 5]})
    return models


# ----------------------------------------------------------------------------
# 3. EVALUATION HELPERS
# ----------------------------------------------------------------------------
def metrics_at(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), Specificity=tn / max(tn + fp, 1), F1=f1_score(y, pred),
                ROC_AUC=roc_auc_score(y, proba), PR_AUC=average_precision_score(y, proba),
                Brier=brier_score_loss(y, proba), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def pick_threshold(y, proba, target_recall):
    """Highest threshold whose sensitivity on (out-of-fold) training data still meets the target."""
    ts = np.unique(np.round(proba, 4))[::-1]
    for t in ts:
        if recall_score(y, (proba >= t).astype(int)) >= target_recall:
            return float(t)
    return 0.5


def plot_roc(curves, path, title):
    plt.figure(figsize=(6.5, 5.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate (1 - specificity)"); plt.ylabel("True positive rate (sensitivity)")
    plt.title(title); plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def plot_confusions(y, proba, thr, name, label, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, t, ttl in [(ax[0], 0.5, "threshold 0.50"), (ax[1], thr, f"screening threshold {thr:.2f}")]:
        cm = confusion_matrix(y, (proba >= t).astype(int), labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["healthy", label], yticklabels=["healthy", label],
              xlabel="Predicted", ylabel="Actual", title=ttl)
    plt.suptitle(f"{name} - test set"); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_importance(imp, path, name):
    top = imp.sort_values().tail(15)
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values, color="#2b6cb0")
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title(f"Permutation importance - {name}"); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
def predict_mode(args):
    bundle = joblib.load(Path(args.out_dir) / f"{args.dataset}_model.joblib")
    X = pd.read_csv(args.predict, na_values=["?", "NA", ""])
    X = X.reindex(columns=bundle["features"])
    p = bundle["model"].predict_proba(X)[:, 1]
    print(f"Model: {bundle['name']} | screening threshold {bundle['threshold']:.2f}")
    for i, pr in enumerate(p):
        flag = f"HIGH RISK of {bundle['label']}" if pr >= bundle["threshold"] else "lower risk"
        print(f"  patient {i}: probability {pr:.1%} -> {flag}")
    print("Educational model only - not a medical diagnosis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "heart", "diabetes", "custom"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default="target")
    ap.add_argument("--out_dir", default="outputs_disease")
    ap.add_argument("--n_iter", type=int, default=15, help="random-search trials per model")
    ap.add_argument("--target_recall", type=float, default=TARGET_SENSITIVITY)
    ap.add_argument("--predict", default=None, help="CSV of new patients (same columns) to score")
    args, _ = ap.parse_known_args()
    Path(args.out_dir).mkdir(exist_ok=True)
    if args.predict:
        return predict_mode(args)

    t0 = time.time()
    X, y, label = load_data(args)
    print(f"Dataset: {args.dataset} | {X.shape[0]} patients, {X.shape[1]} features | "
          f"{label}: {y.mean():.1%} positive")
    print(f"Missing values: {int(X.isna().sum().sum())} | Boosting model: {'XGBoost' if HAVE_XGB else 'sklearn HistGradientBoosting (xgboost not installed)'}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted, rows, test_proba, oof = {}, [], {}, {}
    for name, (pipe, space) in build_models(Xtr, ytr).items():
        n_combos = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(pipe, space, n_iter=min(args.n_iter, n_combos), scoring="roc_auc", cv=cv,
                                    random_state=SEED, n_jobs=2, refit=True)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        fitted[name] = best
        test_proba[name] = best.predict_proba(Xte)[:, 1]
        oof[name] = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        r = metrics_at(yte, test_proba[name])
        r = {"Model": name, "CV_ROC_AUC": search.best_score_, **r}
        rows.append(r)
        print(f"  tuned {name:38s} CV AUC {search.best_score_:.4f} | test AUC {r['ROC_AUC']:.4f}  {search.best_params_}")

    res = pd.DataFrame(rows).set_index("Model")
    show = ["CV_ROC_AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"]
    print("\n=== Test-set performance at threshold 0.50 ===")
    print(res[show].round(3).to_string())
    res.round(4).to_csv(Path(args.out_dir) / f"{args.dataset}_model_comparison.csv")

    best_name = res["CV_ROC_AUC"].idxmax()           # choose by cross-validation, never by the test set
    thr = pick_threshold(ytr, oof[best_name], args.target_recall)
    m = metrics_at(yte, test_proba[best_name], thr)
    print(f"\nBest model by cross-validated ROC-AUC: {best_name}")
    print(f"Screening threshold for >= {args.target_recall:.0%} sensitivity (chosen on training folds): {thr:.2f}")
    print(f"  at threshold {thr:.2f}: sensitivity {m['Recall']:.3f} | specificity {m['Specificity']:.3f} | "
          f"precision {m['Precision']:.3f} | missed cases (FN) {m['FN']} | false alarms (FP) {m['FP']}")

    plot_roc({k: (yte, v) for k, v in test_proba.items()}, Path(args.out_dir) / f"{args.dataset}_roc.png",
             f"ROC curves - {label} prediction (test set)")
    plot_confusions(yte, test_proba[best_name], thr, best_name, label, Path(args.out_dir) / f"{args.dataset}_confusion.png")
    pi = permutation_importance(fitted[best_name], Xte, yte, scoring="roc_auc", n_repeats=20,
                                random_state=SEED, n_jobs=2)
    imp = pd.Series(pi.importances_mean, index=X.columns)
    plot_importance(imp, Path(args.out_dir) / f"{args.dataset}_importance.png", best_name)
    print("\nMost influential features:\n" + imp.sort_values(ascending=False).head(8).round(4).to_string())

    joblib.dump(dict(model=fitted[best_name], name=best_name, threshold=thr, features=list(X.columns),
                     label=label), Path(args.out_dir) / f"{args.dataset}_model.joblib")
    print(f"\nSaved outputs to ./{args.out_dir}/  [{time.time() - t0:.0f}s]")
    print("Educational model only - not a medical diagnosis.")


if __name__ == "__main__":
    main()
