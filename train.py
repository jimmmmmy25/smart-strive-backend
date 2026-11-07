"""
SMART STRIVE - Stress Level Classification (training + evaluation + visualization)

Every block prints a header and saves its charts to <out>/plots/ so you can
inspect each stage of the pipeline.

Usage:
    python train.py --data data/StressLevelDataset.csv --feature-set questionnaire
    python train.py --data data/StressLevelDataset.csv --show      # also open windows
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:          # headless unless you ask to see windows
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, f1_score,
                             precision_recall_fscore_support, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

SEED = 42
TARGET = "stress_level"                 # 0 = low, 1 = moderate, 2 = high
CLASS_NAMES = ["Low", "Moderate", "High"]
COLORS = ["#4C9F70", "#E0A030", "#C0453A"]   # green / amber / red

QUESTIONNAIRE_FEATURES = [
    "study_load", "academic_performance", "future_career_concerns",
    "teacher_student_relationship", "basic_needs", "social_support",
    "peer_pressure", "bullying", "extracurricular_activities",
    "sleep_quality", "living_conditions", "safety", "noise_level",
]

SHOW = "--show" in sys.argv
PLOT_DIR = Path("artifacts/plots")


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=200)
    print(f"  [plot saved] {PLOT_DIR / name}")
    if SHOW:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- BLOCK 1
def load_data(path, feature_set):
    header("BLOCK 1: DATA LOADING & CLEANING")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert TARGET in df.columns, f"'{TARGET}' not found. Columns: {list(df.columns)}"

    n_missing = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())
    print("Shape:", df.shape)
    print("Total missing values:", n_missing)
    print("Duplicate rows:", n_dupes)
    print("Class distribution:\n", df[TARGET].value_counts(normalize=True).sort_index())

    df = df.drop_duplicates().dropna()
    features = (QUESTIONNAIRE_FEATURES if feature_set == "questionnaire"
                else [c for c in df.columns if c != TARGET])
    missing = [f for f in features if f not in df.columns]
    assert not missing, f"Missing columns in dataset: {missing}"

    # Visual: raw data quality (missing per column)
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    fig, ax = plt.subplots(figsize=(8, 4))
    raw.isna().sum().sort_values(ascending=False).head(15).plot.bar(ax=ax, color="#7A8CA5")
    ax.set_title("Block 1 - Missing values per column (top 15)")
    ax.set_ylabel("count")
    save(fig, "01_missing_values.png")
    return df, features


# ---------------------------------------------------------------- BLOCK 2
def eda(df, features):
    header("BLOCK 2: EXPLORATORY DATA ANALYSIS")
    # 2a class distribution
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
                  color=[COLORS[i] for i in counts.index])
    ax.bar_label(bars)
    ax.set_title("Block 2 - Stress level distribution")
    ax.set_ylabel("students")
    save(fig, "02a_class_distribution.png")

    # 2b feature histograms
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    for ax, f in zip(axes.ravel(), features):
        ax.hist(df[f], bins=min(20, df[f].nunique()), color="#5B7DB1", edgecolor="white")
        ax.set_title(f, fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Block 2 - Feature distributions")
    save(fig, "02b_feature_histograms.png")

    # 2c correlation heatmap (features + target)
    corr = df[features + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Block 2 - Correlation heatmap")
    save(fig, "02c_correlation_heatmap.png")

    # 2d box plots of the 6 features most correlated with stress level
    top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(6).index
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, f in zip(axes.ravel(), top):
        data = [df.loc[df[TARGET] == c, f] for c in range(3)]
        bp = ax.boxplot(data, tick_labels=CLASS_NAMES, patch_artist=True)
        for patch, col in zip(bp["boxes"], COLORS):
            patch.set_facecolor(col)
        ax.set_title(f, fontsize=9)
    fig.suptitle("Block 2 - Top correlated features by stress level")
    save(fig, "02d_boxplots_top_features.png")


# ---------------------------------------------------------------- BLOCK 3
def get_candidates():
    cands = {
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced"),
            {"n_estimators": [200, 300, 500], "max_depth": [None, 6, 10, 15],
             "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
             "max_features": ["sqrt", "log2", 0.5]},
        )
    }
    if XGBClassifier is not None:
        cands["XGBoost"] = (
            XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 300, 500], "max_depth": [3, 4, 6],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 3, 5],
             "reg_lambda": [1, 3, 5]},
        )
    return cands


def tune_models(X_train, y_train, cv):
    header("BLOCK 3: TRAINING & HYPERPARAMETER TUNING")
    fitted, searches, cv_fold_scores = {}, {}, {}
    for name, (model, grid) in get_candidates().items():
        print(f"\nTuning {name} ...")
        search = RandomizedSearchCV(model, grid, n_iter=25, cv=cv, scoring="f1_macro",
                                    random_state=SEED, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        searches[name] = search
        cv_fold_scores[name] = cross_val_score(
            search.best_estimator_, X_train, y_train, cv=cv, scoring="f1_macro")
        print("  best params:", search.best_params_)
        print(f"  CV macro-F1: {cv_fold_scores[name].mean():.4f} +/- {cv_fold_scores[name].std():.4f}")

    # Visual 3a: all 25 trials per model (shows how sensitive the model is to settings)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, s in searches.items():
        ax.plot(sorted(s.cv_results_["mean_test_score"], reverse=True), marker="o", label=name)
    ax.set_xlabel("trial rank")
    ax.set_ylabel("CV macro-F1")
    ax.set_title("Block 3 - Hyperparameter search trials")
    ax.legend()
    save(fig, "03a_tuning_trials.png")

    # Visual 3b: fold-by-fold CV comparison
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(list(cv_fold_scores.values()), tick_labels=list(cv_fold_scores.keys()))
    ax.set_ylabel("CV macro-F1 (5 folds)")
    ax.set_title("Block 3 - Cross-validation comparison")
    save(fig, "03b_cv_comparison.png")
    return fitted, cv_fold_scores


# ---------------------------------------------------------------- BLOCK 4
def evaluate(fitted, cv_fold_scores, X_test, y_test):
    header("BLOCK 4: EVALUATION ON HELD-OUT TEST SET")
    results = {}
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    for name, model in fitted.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        results[name] = {
            "cv_f1_macro_mean": float(cv_fold_scores[name].mean()),
            "cv_f1_macro_std": float(cv_fold_scores[name].std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
        }
        print(f"\n--- {name} ---")
        print(classification_report(y_test, pred, target_names=CLASS_NAMES))

        # 4a confusion matrices (counts + row-normalised)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=CLASS_NAMES, cmap="Blues", ax=axes[0])
        axes[0].set_title("Counts")
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=CLASS_NAMES, cmap="Blues",
            normalize="true", values_format=".2f", ax=axes[1])
        axes[1].set_title("Normalised (recall per class)")
        fig.suptitle(f"Block 4 - {name} confusion matrix")
        save(fig, f"04a_confusion_{name}.png")

        # 4b ROC curves, one-vs-rest
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for i, cname in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, color=COLORS[i], label=f"{cname} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"Block 4 - {name} ROC (one-vs-rest)")
        ax.legend()
        save(fig, f"04b_roc_{name}.png")

        # 4c per-class precision / recall / F1
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1, 2])
        x = np.arange(3)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - 0.25, p, 0.25, label="Precision")
        ax.bar(x, r, 0.25, label="Recall")
        ax.bar(x + 0.25, f, 0.25, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Block 4 - {name} per-class metrics")
        ax.legend(loc="lower right")
        save(fig, f"04c_per_class_{name}.png")

    # 4d side-by-side model comparison
    names = list(results)
    metrics = ["cv_f1_macro_mean", "test_accuracy", "test_f1_macro"]
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, n in enumerate(names):
        vals = [results[n][m] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=n)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(["CV macro-F1", "Test accuracy", "Test macro-F1"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Block 4 - Model comparison")
    ax.legend()
    save(fig, "04d_model_comparison.png")
    return results


# ---------------------------------------------------------------- BLOCK 5
def overfitting_check(best_model, best_name, X_train, y_train, cv):
    header("BLOCK 5: OVERFITTING / DATA-SIZE CHECK (learning curve)")
    sizes, tr, va = learning_curve(best_model, X_train, y_train, cv=cv,
                                   scoring="f1_macro", train_sizes=np.linspace(0.2, 1, 6),
                                   n_jobs=-1, random_state=SEED, shuffle=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, tr.mean(1), "o-", label="Training score")
    ax.plot(sizes, va.mean(1), "o-", label="Validation score")
    ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.15)
    ax.set_xlabel("training samples")
    ax.set_ylabel("macro-F1")
    ax.set_title(f"Block 5 - {best_name} learning curve")
    ax.legend()
    save(fig, "05_learning_curve.png")
    gap = tr.mean(1)[-1] - va.mean(1)[-1]
    print(f"  train-validation gap at full size: {gap:.3f} (large gap => overfitting)")


# ---------------------------------------------------------------- BLOCK 6
def interpret(best_model, best_name, X_test, y_test, features):
    header("BLOCK 6: MODEL INTERPRETATION (which factors drive stress?)")
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=15,
                                  random_state=SEED, scoring="f1_macro", n_jobs=-1)
    order = np.argsort(perm.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(features)[order], perm.importances_mean[order],
            xerr=perm.importances_std[order], color="#5B7DB1")
    ax.set_xlabel("drop in macro-F1 when feature is shuffled")
    ax.set_title(f"Block 6 - {best_name} permutation importance")
    save(fig, "06a_permutation_importance.png")
    imp = pd.Series(perm.importances_mean, index=features)

    # Optional SHAP (only if installed: pip install shap)
    try:
        import shap
        sv = shap.TreeExplainer(best_model).shap_values(X_test)
        arr = np.abs(np.array(sv))
        feat_axis = [i for i, s in enumerate(arr.shape) if s == len(features)][-1]
        mean_abs = arr.mean(axis=tuple(i for i in range(arr.ndim) if i != feat_axis))
        order = np.argsort(mean_abs)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(features)[order], mean_abs[order], color="#8B5FBF")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title(f"Block 6 - {best_name} SHAP importance")
        save(fig, "06b_shap_importance.png")
    except ImportError:
        print("  (shap not installed - skipping SHAP plot)")
    except Exception as e:  # SHAP output shapes vary by version
        print(f"  (SHAP skipped: {e})")
    return imp


# ---------------------------------------------------------------- MAIN
def main(args):
    global PLOT_DIR
    out = Path(args.out)
    PLOT_DIR = out / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df, features = load_data(args.data, args.feature_set)        # Block 1
    eda(df, features)                                            # Block 2
    X, y = df[features], df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fitted, cv_scores = tune_models(X_train, y_train, cv)        # Block 3
    results = evaluate(fitted, cv_scores, X_test, y_test)        # Block 4

    # select on CV score so the test set stays untouched for final reporting
    best_name = max(results, key=lambda k: results[k]["cv_f1_macro_mean"])
    best_model = fitted[best_name]
    print(f"\nSelected model: {best_name}")

    overfitting_check(best_model, best_name, X_train, y_train, cv)  # Block 5
    imp = interpret(best_model, best_name, X_test, y_test, features)  # Block 6

    header("SAVING ARTIFACTS")
    joblib.dump(best_model, out / "stress_model.joblib")
    meta = {
        "model": best_name, "feature_set": args.feature_set,
        "features": features, "classes": CLASS_NAMES,
        "feature_ranges": {f: [float(X[f].min()), float(X[f].max())] for f in features},
        "feature_importance": {k: float(v) for k, v in imp.sort_values(ascending=False).items()},
        "results": results,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model + metadata to {out}/ and charts to {PLOT_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--feature-set", choices=["questionnaire", "full"], default="questionnaire")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--show", action="store_true", help="also display plot windows")
    main(p.parse_args())
