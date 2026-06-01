from pathlib import Path
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data_bert128.csv"
REPORT_PATH = BASE_DIR / "single_ros_128_classification_report.txt"
RANDOM_STATE = 42
N_SPLITS = 5
POSITIVE_LABEL = 1
LABEL_MAP = {"F": 0, "T": 1}
INVERSE_LABEL_MAP = {0: "F", 1: "T"}
REFIT_METRIC = "f1"


def make_pipeline(classifier, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    steps.append(("ros", RandomOverSampler(sampling_strategy="auto", random_state=RANDOM_STATE)))
    steps.append(("scaler", StandardScaler() if scale else "passthrough"))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def make_search_spaces() -> dict[str, list[dict]]:
    return {
        "lgbm": [
            {
                "classifier": [
                    LGBMClassifier(
                        objective="binary",
                        n_jobs=4,
                        verbosity=-1,
                        random_state=RANDOM_STATE,
                    )
                ],
                "classifier__n_estimators": [240, 300, 360],
                "classifier__learning_rate": [0.02, 0.03],
                "classifier__num_leaves": [15, 31],
                "classifier__max_depth": [3, 5, 8],
                "classifier__min_child_samples": [20, 30],
                "classifier__subsample": [0.85],
                "classifier__colsample_bytree": [0.85],
                "classifier__reg_lambda": [1.0, 2.0],
            }
        ],
        "xgb": [
            {
                "classifier": [
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    )
                ],
                "classifier__n_estimators": [300, 360, 420],
                "classifier__learning_rate": [0.03, 0.05, 0.07],
                "classifier__max_depth": [2, 3, 4],
                "classifier__min_child_weight": [2, 4, 6],
                "classifier__subsample": [0.85],
                "classifier__colsample_bytree": [0.85],
                "classifier__reg_lambda": [3.0, 5.0, 7.0],
            }
        ],
        "svm": [
            {
                "scaler": [StandardScaler()],
                "classifier": [
                    SVC(
                        probability=True,
                        random_state=RANDOM_STATE,
                    )
                ],
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.5, 1.0, 2.0, 3.0, 4.0],
                "classifier__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
            },
            {
                "scaler": [StandardScaler()],
                "classifier": [
                    SVC(
                        probability=True,
                        random_state=RANDOM_STATE,
                    )
                ],
                "classifier__kernel": ["linear"],
                "classifier__C": [0.1, 0.3, 1.0, 3.0, 4.0],
            },
            {
                "scaler": [StandardScaler()],
                "classifier": [
                    SVC(
                        probability=True,
                        random_state=RANDOM_STATE,
                    )
                ],
                "classifier__kernel": ["poly"],
                "classifier__C": [0.1, 0.3, 1.0, 2.0, 3.0],
                "classifier__degree": [2, 3],
                "classifier__gamma": ["scale", 0.01],
                "classifier__coef0": [0.0, 1.0],
            },
        ],
        "random_forest": [
            {
                "classifier": [
                    RandomForestClassifier(
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    )
                ],
                "classifier__n_estimators": [300, 500, 700],
                "classifier__max_depth": [8, 16, 64],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__max_features": ["sqrt", 0.5],
            }
        ],
        "logistic_regression": [
            {
                "scaler": [StandardScaler()],
                "classifier": [
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    )
                ],
                "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0],
                "classifier__l1_ratio": [0.0, 1.0],
            }
        ],
    }


def get_positive_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    positive_index = list(model.classes_).index(POSITIVE_LABEL)
    return model.predict_proba(x)[:, positive_index]


def cross_validated_scores(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.empty(len(y), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        oof_scores[valid_idx] = get_positive_scores(fold_model, x.iloc[valid_idx])
        print(f"OOF fold {fold}/{N_SPLITS} done")

    return oof_scores


def find_best_threshold(
    y_true: pd.Series,
    positive_scores: np.ndarray,
) -> tuple[float, float, float]:
    candidate_thresholds = np.unique(np.quantile(positive_scores, np.linspace(0.01, 0.99, 197)))
    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = -1.0

    for threshold in candidate_thresholds:
        pred = (positive_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_true, pred)
        macro_f1 = f1_score(y_true, pred, average="macro")

        if macro_f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(macro_f1)
            best_accuracy = float(accuracy)

    return best_threshold, best_accuracy, best_f1


def summarize_params(params: dict) -> dict:
    summary = {}
    for key, value in params.items():
        clean_key = key.replace("classifier__", "")
        if key == "classifier":
            summary["classifier"] = value.__class__.__name__
        elif key == "scaler":
            summary["scaler"] = value.__class__.__name__
        else:
            summary[clean_key] = value
    return summary


def grid_candidate_count(param_grid: list[dict]) -> int:
    return sum(np.prod([len(values) for values in params.values()]) for params in param_grid)


def run_family_grid(
    family_name: str,
    param_grid: list[dict],
    base_pipeline: Pipeline,
    scoring: dict,
    cv: StratifiedKFold,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[GridSearchCV, pd.DataFrame]:
    candidates = grid_candidate_count(param_grid)
    print(f"\n===== Grid search: {family_name.upper()} ({candidates} candidates) =====")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=REFIT_METRIC,
        n_jobs=1,
        verbose=1,
    )
    grid_search.fit(x, y)

    results = pd.DataFrame(grid_search.cv_results_)
    results["family"] = family_name
    results["model"] = results["param_classifier"].map(lambda estimator: estimator.__class__.__name__)
    results["short_params"] = results["params"].map(summarize_params)

    print(f"Best {family_name.upper()} CV macro-F1: {grid_search.best_score_:.4f}")
    print("Best parameters:")
    for key, value in grid_search.best_params_.items():
        if key in {"classifier", "scaler"}:
            value = value.__class__.__name__
        print(f"  {key}: {value}")

    return grid_search, results


def make_estimator_from_params(base_pipeline: Pipeline, params: dict) -> Pipeline:
    estimator = clone(base_pipeline)
    estimator.set_params(**params)
    return estimator


def evaluate_single_model(
    family_name: str,
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict:
    print(f"\n===== OOF evaluation: {family_name.upper()} =====")
    oof_scores = cross_validated_scores(model, x, y)
    threshold, accuracy, macro_f1 = find_best_threshold(y, oof_scores)
    oof_pred = (oof_scores >= threshold).astype(int)
    auc = roc_auc_score(y, oof_scores)
    report_text = classification_report(
        y,
        oof_pred,
        target_names=[INVERSE_LABEL_MAP[0], INVERSE_LABEL_MAP[1]],
        digits=4,
    )

    print("OOF threshold-tuned evaluation:")
    print(f"Best threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print()
    print(report_text)

    return {
        "family": family_name,
        "threshold": threshold,
        "accuracy": accuracy,
        "auc": auc,
        "macro_f1": macro_f1,
        "report_text": report_text,
    }


def write_best_report(best_result: dict, best_params: dict, comparison: pd.DataFrame) -> None:
    lines = [
        f"Best model: {best_result['family']}",
        f"Best threshold: {best_result['threshold']:.6f}",
        f"Accuracy: {best_result['accuracy']:.6f}",
        f"AUC: {best_result['auc']:.6f}",
        f"Macro-F1: {best_result['macro_f1']:.6f}",
        "",
        "Best parameters:",
    ]
    for key, value in summarize_params(best_params).items():
        lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "Model comparison:",
            comparison.to_string(index=False),
            "",
            "Classification report:",
            best_result["report_text"],
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise KeyError("training_data_bert128.csv must contain a label column")

    y = df["label"].map(LABEL_MAP)
    if y.isna().any():
        bad_labels = sorted(df.loc[y.isna(), "label"].dropna().unique())
        raise ValueError(f"Unexpected label values: {bad_labels}")
    y = y.astype(int)
    x = df.drop(columns=["label"])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    base_pipeline = make_pipeline(LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1))
    search_spaces = make_search_spaces()
    grid_candidates = sum(grid_candidate_count(param_grid) for param_grid in search_spaces.values())
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro",
        "auc": "roc_auc",
    }

    print(f"Rows: {len(df)}")
    print(f"Features: {x.shape[1]}")
    print("Models: XGBoost, LightGBM, SVM, Random Forest, Logistic Regression")
    print(f"Grid candidates: {grid_candidates}")
    print(f"Cross-validation: StratifiedKFold, {N_SPLITS} folds")
    print(f"Refit metric: {REFIT_METRIC} (macro-F1)")
    print("Imbalance handling: RandomOverSampler only")
    print("Saved model: disabled")
    print()

    family_results = []
    best_params_by_family = {}
    best_models_by_family = {}

    for family_name, param_grid in search_spaces.items():
        grid_search, results = run_family_grid(
            family_name,
            param_grid,
            base_pipeline,
            scoring,
            cv,
            x,
            y,
        )
        family_results.append(results)
        best_params_by_family[family_name] = grid_search.best_params_
        best_models_by_family[family_name] = make_estimator_from_params(
            base_pipeline,
            grid_search.best_params_,
        )

    all_results = pd.concat(family_results, ignore_index=True)
    print("\n===== Best CV results by family =====")
    cv_summary = (
        all_results.sort_values(["family", f"rank_test_{REFIT_METRIC}"])
        .groupby("family", as_index=False)
        .head(1)
        [
            [
                "family",
                "model",
                "mean_test_accuracy",
                "mean_test_f1",
                "mean_test_auc",
                f"std_test_{REFIT_METRIC}",
                "short_params",
            ]
        ]
        .sort_values("mean_test_f1", ascending=False)
    )
    print(cv_summary.to_string(index=False))

    evaluated_results = []
    for family_name, model in best_models_by_family.items():
        evaluated_results.append(evaluate_single_model(family_name, model, x, y))

    comparison = pd.DataFrame(
        [
            {
                "family": result["family"],
                "threshold": result["threshold"],
                "accuracy": result["accuracy"],
                "auc": result["auc"],
                "macro_f1": result["macro_f1"],
            }
            for result in evaluated_results
        ]
    ).sort_values("macro_f1", ascending=False)

    best_result = max(evaluated_results, key=lambda result: result["macro_f1"])
    write_best_report(
        best_result,
        best_params_by_family[best_result["family"]],
        comparison,
    )

    print("\n===== Final single-model selection =====")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_result['family']}")
    print(f"Classification report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
