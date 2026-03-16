from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


from inference.predictor import LimiXPredictor


DATA_PATH = Path("data.csv")
MODEL_PATH = Path("modelCache/LimiX-2M.ckpt")
GPU_CONFIG_PATH = Path("config/reg_default_2M_retrieval.json")
CPU_CONFIG_PATH = Path("config/reg_default_noretrieval.json")
OUTPUT_DIR = Path("outputs") / "pm25_regression"

CATEGORICAL_COLUMNS = ["PROVINCE", "CITY", "COUNTY"]
NUMERICAL_COLUMNS = ["AET", "ppt", "tem", "wind", "NOX", "SO2", "fertilzier", "manure"]
TARGET_COLUMN = "PM2.5"


def parse_args() -> argparse.Namespace:
    """Provide a few practical overrides while keeping the default run simple."""
    parser = argparse.ArgumentParser(
        description="Use LimiX to predict PM2.5 and export evaluation plots."
    )
    parser.add_argument(
        "--data-path", type=Path, default=DATA_PATH, help="CSV dataset path."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="LimiX model checkpoint path.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional inference config path. If omitted, a default config is selected based on the device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for metrics, CSV and plots.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio.")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for train/test split."
    )
    return parser.parse_args()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop samples with missing values to keep the inference input consistent."""
    return df.dropna().copy()


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the CSV file and validate the columns needed by the regression pipeline."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")

    try:
        df = pd.read_csv(data_path, engine="pyarrow")
    except Exception:
        # Fallback to the default parser when pyarrow is unavailable.
        df = pd.read_csv(data_path)

    df = clean_data(df)
    required_columns = set(CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("The dataset is empty after dropping missing values.")

    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to integer ids before feeding them to the model."""
    encoded_df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        encoded_df[column] = (
            encoded_df[column].astype("category").cat.codes.astype(np.float32)
        )
    return encoded_df


def build_features_and_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the model input matrix X and target vector y."""
    feature_columns = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
    encoded_df = encode_categorical_columns(df)
    x = encoded_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    y = encoded_df.loc[:, TARGET_COLUMN].to_numpy(dtype=np.float32)
    return x, y


def normalize_target(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize the training target to match the regression example in the repo."""
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_std = max(y_std, 1e-8)
    y_train_norm = (y_train - y_mean) / y_std
    return y_train_norm, y_mean, y_std


def resolve_inference_config(
    device: torch.device, custom_config_path: Path | None = None
) -> Path:
    """Use a user-specified config if provided, otherwise fall back to a device-safe default."""
    config_path = (
        custom_config_path
        if custom_config_path is not None
        else (GPU_CONFIG_PATH if device.type == "cuda" else CPU_CONFIG_PATH)
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Inference config does not exist: {config_path}")
    return config_path


def create_predictor(
    device: torch.device, model_path: Path, config_path: Path
) -> LimiXPredictor:
    """Create the LimiX predictor with the categorical column positions declared explicitly."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    return LimiXPredictor(
        device=device,
        model_path=str(model_path),
        inference_config=str(config_path),
        categorical_features_indices=[0, 1, 2],
    )


def to_numpy(array_like) -> np.ndarray:
    """Unify numpy and torch outputs for downstream metrics and plotting."""
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=np.float32).reshape(-1)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the core regression metrics used in the original script."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_prediction_results(
    output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray
) -> Path:
    """Save the prediction table for later inspection."""
    residual = y_true - y_pred
    results_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
        }
    )
    results_path = output_dir / "prediction_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    return results_path


def plot_fit_curve(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Path:
    """Plot the fitted curves of true and predicted values after sorting by the true target."""
    sorted_idx = np.argsort(y_true)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true_sorted, label="True PM2.5", linewidth=2.2, color="#1f77b4")
    ax.plot(
        y_pred_sorted,
        label="Predicted PM2.5",
        linewidth=2.0,
        color="#ff7f0e",
        alpha=0.9,
    )
    ax.set_title("Prediction vs True Fit Curve")
    ax.set_xlabel("Samples sorted by true value")
    ax.set_ylabel("PM2.5")
    ax.grid(alpha=0.25)
    ax.legend()

    figure_path = output_dir / "prediction_fit_curve.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_residual_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path
) -> Path:
    """Plot the residual histogram to show the prediction error distribution."""
    residual = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=30, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="#c44e52", linestyle="--", linewidth=2, label="Zero Residual")
    ax.axvline(
        float(residual.mean()),
        color="#55a868",
        linestyle="-.",
        linewidth=2,
        label=f"Mean Residual = {residual.mean():.2f}",
    )
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (True - Predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    ax.legend()

    figure_path = output_dir / "residual_distribution.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run_regression_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Execute the end-to-end PM2.5 regression workflow."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_path)
    x, y = build_features_and_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    y_train_norm, y_mean, y_std = normalize_target(y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = resolve_inference_config(device, args.config_path)
    predictor = create_predictor(
        device=device, model_path=args.model_path, config_path=config_path
    )

    y_pred_norm = predictor.predict(
        x_train, y_train_norm, x_test, task_type="Regression"
    )
    y_pred = to_numpy(y_pred_norm) * y_std + y_mean

    metrics = evaluate_predictions(y_test, y_pred)
    results_path = save_prediction_results(output_dir, y_test, y_pred)
    fit_curve_path = plot_fit_curve(y_test, y_pred, output_dir)
    residual_plot_path = plot_residual_distribution(y_test, y_pred, output_dir)

    return {
        "device": device.type,
        "config_path": config_path,
        "num_samples": len(df),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "metrics": metrics,
        "results_path": results_path,
        "fit_curve_path": fit_curve_path,
        "residual_plot_path": residual_plot_path,
    }


def main() -> None:
    args = parse_args()
    summary = run_regression_pipeline(args)

    print(f"Device: {summary['device']}")
    print(f"Inference config: {summary['config_path']}")
    print(f"Samples after cleaning: {summary['num_samples']}")
    print(f"Train/Test split: {summary['train_size']} / {summary['test_size']}")
    print(f"RMSE: {summary['metrics']['rmse']:.4f}")
    print(f"R2: {summary['metrics']['r2']:.4f}")
    print(f"Prediction table saved to: {summary['results_path']}")
    print(f"Fit curve saved to: {summary['fit_curve_path']}")
    print(f"Residual distribution saved to: {summary['residual_plot_path']}")


if __name__ == "__main__":
    main()
