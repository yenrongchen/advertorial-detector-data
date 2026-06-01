from pathlib import Path
import itertools
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data.csv"
MODEL_PATH = BASE_DIR / "advertorial_deep_classifier.pt"
RANDOM_STATE = 42
N_SPLITS = 5
POSITIVE_LABEL = 1
LABEL_MAP = {"F": 0, "T": 1}
INVERSE_LABEL_MAP = {0: "F", 1: "T"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CONFIGS = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.transform(x)
        gate = self.gate(x)
        return self.activation(x + gate * transformed)


class TabularResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        block_type: str,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        block_cls = GatedResidualBlock if block_type == "gated" else ResidualBlock
        self.blocks = nn.Sequential(*[block_cls(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x).squeeze(1)


def make_param_grid() -> list[dict]:
    # grid = {
    #     "hidden_dim": [64, 128],
    #     "num_blocks": [2, 3, 4],
    #     "dropout": [0.10, 0.20, 0.30],
    #     "lr": [5e-3, 1e-3, 5e-4],
    #     "weight_decay": [1e-5, 1e-4],
    #     "batch_size": [32, 64],
    #     "pos_weight_multiplier": [1.0, 1.2, 1.5],
    #     "block_type": ["gated"],
    #     "max_epochs": [300],
    #     "patience": [50],
    # }

    grid = {
        "hidden_dim": [64, 128],
        "num_blocks": [3, 4],
        "dropout": [0.10, 0.20],
        "lr": [3e-3, 1e-3],
        "weight_decay": [1e-5],
        "batch_size": [32, 64],
        "pos_weight_multiplier": [1.5, 2, 2.5],
        "block_type": ["gated"],
        "max_epochs": [300],
        "patience": [35],
    }
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def prepare_xy() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise KeyError("training_data.csv must contain a label column")

    y = df["label"].map(LABEL_MAP)
    if y.isna().any():
        bad_labels = sorted(df.loc[y.isna(), "label"].dropna().unique())
        raise ValueError(f"Unexpected label values: {bad_labels}")

    x = df.drop(columns=["label"])
    return x, y.astype(int).to_numpy()


def preprocess_fold(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(imputer.fit_transform(x_train)).astype(np.float32)
    x_valid_np = scaler.transform(imputer.transform(x_valid)).astype(np.float32)
    return x_train_np, x_valid_np, imputer, scaler


def make_loader(
    x_np: np.ndarray,
    y_np: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x_np.astype(np.float32)),
        torch.from_numpy(y_np.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_fold(
    x_train_np: np.ndarray,
    y_train: np.ndarray,
    x_valid_np: np.ndarray,
    y_valid: np.ndarray,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    model = TabularResidualMLP(
        input_dim=x_train_np.shape[1],
        hidden_dim=params["hidden_dim"],
        num_blocks=params["num_blocks"],
        dropout=params["dropout"],
        block_type=params["block_type"],
    ).to(DEVICE)

    neg_count = max(1, int((y_train == 0).sum()))
    pos_count = max(1, int((y_train == 1).sum()))
    pos_weight = torch.tensor(
        [neg_count / pos_count * params["pos_weight_multiplier"]],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(3, params["patience"] // 3),
    )

    train_loader = make_loader(x_train_np, y_train, params["batch_size"], shuffle=True)
    x_valid_tensor = torch.from_numpy(x_valid_np).to(DEVICE)

    best_state = None
    best_valid_f2 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_prob = torch.sigmoid(model(x_valid_tensor)).cpu().numpy()

        valid_pred = (valid_prob >= 0.5).astype(int)
        valid_f2 = fbeta_score(
            y_valid,
            valid_pred,
            beta=2,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        )
        scheduler.step(valid_f2)

        if valid_f2 > best_valid_f2:
            best_valid_f2 = valid_f2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= params["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        valid_prob = torch.sigmoid(model(x_valid_tensor)).cpu().numpy()

    return valid_prob, {"best_epoch": best_epoch, "best_valid_f2": best_valid_f2}


def find_best_threshold(y_true: np.ndarray, positive_scores: np.ndarray) -> tuple[float, float]:
    candidate_thresholds = np.unique(np.quantile(positive_scores, np.linspace(0.01, 0.99, 197)))
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidate_thresholds:
        pred = (positive_scores >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=2, pos_label=POSITIVE_LABEL, zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def threshold_report(y_true: np.ndarray, positive_scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        pred = (positive_scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, pred),
                "macro_f1": f1_score(y_true, pred, average="macro"),
                "t_precision": precision_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0),
                "t_recall": recall_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0),
                "t_f2": fbeta_score(y_true, pred, beta=2, pos_label=POSITIVE_LABEL, zero_division=0),
                "predicted_t": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_params(x: pd.DataFrame, y: np.ndarray, params: dict) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.empty(len(y), dtype=np.float32)
    fold_epochs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        x_train_np, x_valid_np, _, _ = preprocess_fold(x.iloc[train_idx], x.iloc[valid_idx])
        valid_prob, fold_info = train_one_fold(
            x_train_np,
            y[train_idx],
            x_valid_np,
            y[valid_idx],
            params,
            seed=RANDOM_STATE + fold,
        )
        oof_scores[valid_idx] = valid_prob
        fold_epochs.append(fold_info["best_epoch"])
        print(
            f"  fold {fold}/{N_SPLITS} done | "
            f"best_epoch={fold_info['best_epoch']} | "
            f"valid_f2@0.5={fold_info['best_valid_f2']:.4f}"
        )

    threshold, best_f2 = find_best_threshold(y, oof_scores)
    pred = (oof_scores >= threshold).astype(int)
    return {
        "params": params,
        "threshold": threshold,
        "t_f2": best_f2,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "t_precision": precision_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "t_recall": recall_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "mean_best_epoch": float(np.mean(fold_epochs)),
        "oof_scores": oof_scores,
    }


def train_final_model(
    x: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    final_epochs: int,
) -> dict:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_np = scaler.fit_transform(imputer.fit_transform(x)).astype(np.float32)

    set_seed(RANDOM_STATE)
    model = TabularResidualMLP(
        input_dim=x_np.shape[1],
        hidden_dim=params["hidden_dim"],
        num_blocks=params["num_blocks"],
        dropout=params["dropout"],
        block_type=params["block_type"],
    ).to(DEVICE)

    neg_count = max(1, int((y == 0).sum()))
    pos_count = max(1, int((y == 1).sum()))
    pos_weight = torch.tensor(
        [neg_count / pos_count * params["pos_weight_multiplier"]],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loader = make_loader(x_np, y, params["batch_size"], shuffle=True)

    model.train()
    for _ in range(max(1, final_epochs)):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "imputer": imputer,
        "scaler": scaler,
    }


def main() -> None:
    print(f"Device: {DEVICE}")
    x, y = prepare_xy()
    param_grid = make_param_grid()
    if MAX_CONFIGS is not None:
        param_grid = param_grid[:MAX_CONFIGS]

    print(f"Rows: {len(x)}")
    print(f"Features: {x.shape[1]}")
    print(f"Grid candidates: {len(param_grid)}")
    print(f"Cross-validation: StratifiedKFold, {N_SPLITS} folds")
    print("Refit metric: t_f2")
    print()

    results = []
    for index, params in enumerate(param_grid, start=1):
        print(f"Evaluating config {index}/{len(param_grid)}: {params}")
        result = cross_validate_params(x, y, params)
        results.append(result)
        print(
            f"  OOF threshold={result['threshold']:.4f} | "
            f"t_f2={result['t_f2']:.4f} | "
            f"t_recall={result['t_recall']:.4f} | "
            f"t_precision={result['t_precision']:.4f} | "
            f"macro_f1={result['macro_f1']:.4f}"
        )
        print()

    result_df = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key not in {"oof_scores", "params"}}
            | result["params"]
            for result in results
        ]
    ).sort_values("t_f2", ascending=False)

    print("Top deep model configs:")
    print(result_df.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()

    best_result = max(results, key=lambda result: result["t_f2"])
    best_params = best_result["params"]
    best_threshold = best_result["threshold"]
    best_oof_scores = best_result["oof_scores"]
    best_oof_pred = (best_oof_scores >= best_threshold).astype(int)

    print("Threshold trade-off table:")
    print(threshold_report(y, best_oof_scores).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print("Best deep model params:")
    print(best_params)
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"OOF T F2: {best_result['t_f2']:.4f}")
    print()
    print("Classification Report:")
    print(
        classification_report(
            y,
            best_oof_pred,
            target_names=[INVERSE_LABEL_MAP[0], INVERSE_LABEL_MAP[1]],
            digits=4,
        )
    )

    final_epochs = int(round(best_result["mean_best_epoch"]))
    final_artifacts = train_final_model(x, y, best_params, final_epochs)
    torch.save(
        {
            **final_artifacts,
            "threshold": best_threshold,
            "label_map": LABEL_MAP,
            "inverse_label_map": INVERSE_LABEL_MAP,
            "positive_label": POSITIVE_LABEL,
            "features": list(x.columns),
            "model_class": "TabularResidualMLP",
            "input_dim": x.shape[1],
            "params": best_params,
            "final_epochs": final_epochs,
            "cv_folds": N_SPLITS,
            "cv_results": result_df.to_dict(orient="records"),
            "best_oof_t_f2": best_result["t_f2"],
        },
        MODEL_PATH,
    )
    print(f"Saved deep model trained on all rows: {MODEL_PATH}")


if __name__ == "__main__":
    main()
