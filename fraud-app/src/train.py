import json
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocessing import split_x_y, build_preprocessor, basic_clean
def train(
    data_path: str,
    model_type: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = False,
    class_weight: Optional[str] = "balanced",
    artifacts_dir: str = "artifacts",
):
    df = pd.read_csv(data_path)
    df = basic_clean(df)

    X, y = split_x_y(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pre = build_preprocessor(X)

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight=class_weight, solver="lbfgs")
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=random_state, class_weight=class_weight
        )
    else:
        raise ValueError("model_type must be one of: logreg, rf")

    if use_smote:
        pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=random_state)), ("clf", clf)])
    else:
        pipe = Pipeline([("pre", pre), ("clf", clf)])

    pipe.fit(X_tr, y_tr)

    probas = pipe.predict_proba(X_te)[:, 1]
    preds = (probas >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_te, probas)),
        "pr_auc": float(average_precision_score(y_te, probas)),
        "confusion_matrix": confusion_matrix(y_te, preds).tolist(),
        "report": classification_report(y_te, preds, digits=4, zero_division=0),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "fraud_rate_train": float(y_tr.mean()),
        "fraud_rate_test": float(y_te.mean()),
        "model_type": model_type,
        "use_smote": bool(use_smote),
        "class_weight": class_weight if class_weight else None,
    }

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, artifacts / "pipeline.pkl")
    with open(artifacts / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", artifacts / "pipeline.pkl")
    print("Saved:", artifacts / "metrics.json")
    print("ROC-AUC:", metrics["roc_auc"])
    print("PR-AUC :", metrics["pr_auc"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/creditcard.csv")
    p.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smote", action="store_true")
    p.add_argument("--no-class-weight", action="store_true", help="Disable class_weight")
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()

    cw = None if args.no_class_weight else "balanced"

    train(
        data_path=args.data,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        use_smote=args.smote,
        class_weight=cw,
        artifacts_dir=args.artifacts,
    )



