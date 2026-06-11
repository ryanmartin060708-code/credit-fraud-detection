"""
main.py
-------
CLI entry-point: run the full Credit Card Fraud Detection pipeline
(preprocess → train → evaluate) without launching the Streamlit UI.

Usage
-----
    python main.py
    python main.py --imbalance SMOTE --test-size 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluate import best_model_name, evaluate_all, metrics_dataframe
from src.preprocess import build_pipeline
from src.train import save_best_model, train_all
from src.utils import IMBALANCE_METHODS, get_logger

log = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Full Pipeline"
    )
    parser.add_argument(
        "--imbalance",
        choices=IMBALANCE_METHODS,
        default="SMOTE",
        help="Imbalance handling strategy (default: SMOTE)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (default: 0.20)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Credit Card Fraud Detection Pipeline")
    log.info("=" * 60)
    log.info("Imbalance strategy : %s", args.imbalance)
    log.info("Test fraction      : %.2f", args.test_size)
    log.info("Random state       : %d", args.random_state)
    log.info("-" * 60)

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    log.info("STEP 1 / 3 — Preprocessing")
    pipeline_data = build_pipeline(
        test_size=args.test_size,
        imbalance_method=args.imbalance,
        random_state=args.random_state,
    )

    # ------------------------------------------------------------------
    # Step 2: Train all models
    # ------------------------------------------------------------------
    log.info("STEP 2 / 3 — Training")

    def progress(i, total, name):
        log.info("  [%d/%d] %s", i, total, name)

    use_class_weight = args.imbalance == "Class Weighting"
    models = train_all(
        pipeline_data["X_train"],
        pipeline_data["y_train"],
        class_weight=use_class_weight,
        progress_callback=progress,
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    log.info("STEP 3 / 3 — Evaluation")
    all_metrics = evaluate_all(models, pipeline_data["X_test"], pipeline_data["y_test"])

    # Print comparison table
    df_results = metrics_dataframe(all_metrics).sort_values("ROC AUC", ascending=False)
    log.info("\n%s", df_results.to_string())

    # Save best model
    best_name = best_model_name(all_metrics)
    save_best_model(models[best_name], best_name)

    log.info("=" * 60)
    log.info("✅ Pipeline complete!")
    log.info("   Best model : %s", best_name)
    log.info("   ROC AUC    : %.4f", all_metrics[best_name]["roc_auc"])
    log.info("   F1 Score   : %.4f", all_metrics[best_name]["f1"])
    log.info("=" * 60)
    log.info("Launch the dashboard with:  streamlit run app/Home.py")


if __name__ == "__main__":
    main()
