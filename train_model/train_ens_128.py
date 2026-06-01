from pathlib import Path
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data_bert128.csv"
F1_MODEL_PATH = BASE_DIR / "ensemble_128.joblib"
RANDOM_STATE = 42
N_SPLITS = 5
POSITIVE_LABEL = 1
LABEL_MAP = {"F": 0, "T": 1}
INVERSE_LABEL_MAP = {0: "F", 1: "T"}
REFIT_METRIC = "f1"


def make_pipeline(classifier, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    steps.append(("scaler", StandardScaler() if scale else "passthrough"))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def make_search_spaces(y: pd.Series) -> dict[str, list[dict]]:
    positive_count = max(1, int((y == POSITIVE_LABEL).sum()))
    negative_count = max(1, int((y != POSITIVE_LABEL).sum()))
    positive_weight = negative_count / positive_count
    sqrt_positive_weight = float(np.sqrt(positive_weight))
    boosted_positive_weight = positive_weight * 1.5
    positive_class_weight = {0: 1.0, POSITIVE_LABEL: positive_weight}
    boosted_positive_class_weight = {0: 1.0, POSITIVE_LABEL: boosted_positive_weight}

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
                "classifier__scale_pos_weight": [
                    1.0,
                    sqrt_positive_weight,
                    positive_weight,
                    boosted_positive_weight,
                ],
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
                "classifier__scale_pos_weight": [
                    1.0,
                    sqrt_positive_weight,
                    positive_weight,
                    boosted_positive_weight,
                ],
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
                "classifier__class_weight": [
                    None,
                    "balanced",
                    positive_class_weight,
                    boosted_positive_class_weight,
                ],
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
                "classifier__class_weight": [
                    None,
                    "balanced",
                    positive_class_weight,
                    boosted_positive_class_weight,
                ],
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
                "classifier__class_weight": [
                    None,
                    "balanced",
                    positive_class_weight,
                    boosted_positive_class_weight,
                ],
                "classifier__C": [0.1, 0.3, 1.0, 2.0, 3.0],
                "classifier__degree": [2, 3],
                "classifier__gamma": ["scale", 0.01],
                "classifier__coef0": [0.0, 1.0],
            },
        ],
    }


def get_positive_scores(model, x: pd.DataFrame) -> np.ndarray:
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


def make_ensemble(best_estimators: dict[str, Pipeline]) -> VotingClassifier:
    family_order = ["xgb", "lgbm", "svm"]
    return VotingClassifier(
        estimators=[
            (family_name, best_estimators[family_name])
            for family_name in family_order
            if family_name in best_estimators
        ],
        voting="soft",
        n_jobs=1,
    )


def score_predictions(y_true: pd.Series, pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, pred)
    if metric == "f1":
        return f1_score(y_true, pred, average="macro")
    if metric == "t_f1":
        return f1_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0)
    raise ValueError(f"Unknown threshold metric: {metric}")


def find_best_threshold(
    y_true: pd.Series,
    positive_scores: np.ndarray,
    metric: str,
) -> tuple[float, float, float, float]:
    candidate_thresholds = np.unique(np.quantile(positive_scores, np.linspace(0.01, 0.99, 197)))
    best_threshold = 0.5
    best_score = -1.0
    best_accuracy = -1.0
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        pred = (positive_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, average="macro")
        score = score_predictions(y_true, pred, metric)

        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best_f1 = float(f1)
            best_accuracy = float(accuracy)

    return best_threshold, best_score, best_accuracy, best_f1


def summarize_params(params: dict) -> dict:
    summary = {}
    for key, value in params.items():
        if key == "classifier":
            summary["classifier"] = value.__class__.__name__
        else:
            summary[key.replace("classifier__", "")] = value
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

    print(f"Best {family_name.upper()} CV {REFIT_METRIC}: {grid_search.best_score_:.4f}")
    print("Best parameters:")
    for key, value in grid_search.best_params_.items():
        if key == "classifier":
            value = value.__class__.__name__
        print(f"  {key}: {value}")

    return grid_search, results


def make_estimator_from_params(base_pipeline: Pipeline, params: dict) -> Pipeline:
    estimator = clone(base_pipeline)
    estimator.set_params(**params)
    return estimator


def best_estimators_for_metric(
    metric: str,
    all_results: pd.DataFrame,
    base_pipeline: Pipeline,
) -> tuple[dict[str, Pipeline], dict[str, dict], dict[str, float]]:
    estimators = {}
    params_by_family = {}
    scores_by_family = {}

    for family_name, group in all_results.groupby("family"):
        best_index = group[f"mean_test_{metric}"].idxmax()
        best_row = all_results.loc[best_index]
        params = best_row["params"]
        estimators[family_name] = make_estimator_from_params(base_pipeline, params)
        params_by_family[family_name] = params
        scores_by_family[family_name] = float(best_row[f"mean_test_{metric}"])

    return estimators, params_by_family, scores_by_family


def evaluate_and_save_ensemble(
    model_name: str,
    ensemble_model: VotingClassifier,
    threshold_metric: str,
    output_path: Path,
    x: pd.DataFrame,
    y: pd.Series,
    family_params: dict[str, dict],
    family_scores: dict[str, float],
) -> None:
    print(f"\n===== Evaluating {model_name} ensemble =====")
    print(f"Threshold metric: {threshold_metric}")
    oof_scores = cross_validated_scores(ensemble_model, x, y)
    best_threshold, best_metric_score, best_oof_accuracy, best_oof_f1 = find_best_threshold(
        y,
        oof_scores,
        threshold_metric,
    )
    oof_pred = (oof_scores >= best_threshold).astype(int)
    auc = roc_auc_score(y, oof_scores)
    t_f1 = f1_score(y, oof_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    t_f2 = fbeta_score(y, oof_pred, beta=2, pos_label=POSITIVE_LABEL, zero_division=0)
    report_text = classification_report(
        y,
        oof_pred,
        target_names=[INVERSE_LABEL_MAP[0], INVERSE_LABEL_MAP[1]],
        digits=4,
    )
    report_df = pd.DataFrame(
        classification_report(
            y,
            oof_pred,
            target_names=[INVERSE_LABEL_MAP[0], INVERSE_LABEL_MAP[1]],
            digits=4,
            output_dict=True,
        )
    ).T
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "threshold_metric": threshold_metric,
                "threshold": best_threshold,
                "accuracy": best_oof_accuracy,
                "f1": best_oof_f1,
                "auc": auc,
                "t_f1": t_f1,
                "t_f2": t_f2,
            }
        ]
    )
    report_txt_path = output_path.with_name(f"{output_path.stem}_classification_report.txt")
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.csv")

    print("OOF threshold-tuned evaluation:")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{threshold_metric} score: {best_metric_score:.4f}")
    print(f"Accuracy: {best_oof_accuracy:.4f}")
    print(f"F1: {best_oof_f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"T F1: {t_f1:.4f}")
    print(f"T F2: {t_f2:.4f}")
    print()
    print("Classification Report:")
    print(report_text)

    report_txt_path.write_text(report_text, encoding="utf-8")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    ensemble_model.fit(x, y)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise KeyError("training_data.csv must contain a label column")

    y = df["label"].map(LABEL_MAP)
    if y.isna().any():
        bad_labels = sorted(df.loc[y.isna(), "label"].dropna().unique())
        raise ValueError(f"Unexpected label values: {bad_labels}")
    y = y.astype(int)
    x = df.drop(columns=["label"])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    base_pipeline = make_pipeline(LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1))
    search_spaces = make_search_spaces(y)
    grid_candidates = sum(grid_candidate_count(param_grid) for param_grid in search_spaces.values())
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro",
        "auc": "roc_auc",
        "t_f1": make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "t_recall": make_scorer(recall_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "t_precision": make_scorer(precision_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "t_f2": make_scorer(fbeta_score, beta=2, pos_label=POSITIVE_LABEL, zero_division=0),
    }

    print(f"Rows: {len(df)}")
    print(f"Features: {x.shape[1]}")
    print("Models: XGBoost, LightGBM, SVM")
    print(f"Grid candidates: {grid_candidates}")
    print(f"Cross-validation: StratifiedKFold, {N_SPLITS} folds")
    print(f"Refit metric: {REFIT_METRIC}")
    print("Saved model: macro-F1-selected ensemble with macro-F1 threshold")
    print()

    family_searches = {}
    family_results = []
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
        family_searches[family_name] = grid_search
        family_results.append(results)

    all_results = pd.concat(family_results, ignore_index=True)
    columns = [
        "family",
        "model",
        "mean_test_accuracy",
        "mean_test_f1",
        "mean_test_auc",
        "mean_test_t_f1",
        "mean_test_t_f2",
        "mean_test_t_recall",
        "mean_test_t_precision",
        f"std_test_{REFIT_METRIC}",
        "short_params",
    ]

    f1_estimators, f1_params, f1_scores = best_estimators_for_metric(
        "f1",
        all_results,
        base_pipeline,
    )

    evaluate_and_save_ensemble(
        "f1",
        make_ensemble(f1_estimators),
        "f1",
        F1_MODEL_PATH,
        x,
        y,
        f1_params,
        f1_scores,
    )


if __name__ == "__main__":
    main()
