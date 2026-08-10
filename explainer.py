from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyExplainer:
    """
    Explainer suite for anomaly detection models.
    """

    def __init__(self):
        try:
            import shap
            self.shap = shap
        except ImportError:
            self.shap = None

    def explain_tree_model(
        self,
        model: Any,
        X_background: Any,
        sample: Any,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        SHAP-based explanation for tree models.
        Falls back to a simple perturbation summary if SHAP is unavailable.
        """
        if self.shap is None:
            return {
                "error": "shap is not installed",
            }

        sample = np.asarray(sample)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        try:
            explainer = self.shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            values = np.asarray(shap_values).ravel()

            feature_names = feature_names or [f"feature_{i}" for i in range(len(values))]

            top_idx = np.argsort(np.abs(values))[::-1][:10]

            return {
                "method": "shap_tree",
                "top_features": [
                    {
                        "feature": feature_names[i],
                        "shap_value": float(values[i]),
                        "sample_value": float(sample[0, i]),
                    }
                    for i in top_idx
                ],
            }
        except Exception as exc:
            logger.warning(f"SHAP explanation failed: {exc}")
            return {"method": "shap_tree", "error": str(exc)}

    def gradient_saliency_lstm(
        self,
        model: Any,
        sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Gradient-based saliency map for LSTM autoencoder.

        sequence shape:
        - (seq_len, n_features)
        """
        try:
            import tensorflow as tf
        except ImportError:
            return {"error": "TensorFlow is not installed"}

        seq = np.asarray(sequence, dtype=float)
        if seq.ndim == 2:
            seq = seq[np.newaxis, ...]

        inp = tf.convert_to_tensor(seq, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inp)
            recon = model(inp, training=False)
            loss = tf.reduce_mean(tf.square(inp - recon))

        grads = tape.gradient(loss, inp).numpy()[0]
        saliency = np.mean(np.abs(grads), axis=1)

        top_steps = np.argsort(saliency)[::-1][:10]

        return {
            "method": "gradient_saliency",
            "saliency_by_timestep": saliency.tolist(),
            "top_timesteps": [
                {
                    "timestep": int(step),
                    "saliency": float(saliency[step]),
                }
                for step in top_steps
            ],
        }

    def counterfactual_amount(
        self,
        detector: Any,
        sample: pd.DataFrame | np.ndarray,
        amount_feature: str = "amount",
        step_pct: float = 0.05,
        max_iter: int = 50,
    ) -> Dict[str, Any]:
        """
        Simple counterfactual explanation:
        Reduce amount until the sample is no longer flagged.
        """
        if isinstance(sample, pd.DataFrame):
            base = sample.copy()
        else:
            base = pd.DataFrame(np.asarray(sample).reshape(1, -1))

        if amount_feature not in base.columns:
            return {
                "success": False,
                "message": f"Feature {amount_feature} not found",
            }

        original_amount = float(base[amount_feature].iloc[0])
        current_amount = original_amount

        for _ in range(max_iter):
            current_amount *= 1.0 - step_pct
            candidate = base.copy()
            candidate[amount_feature] = current_amount

            score, label = detector.predict(candidate)

            if label[0] == 0:
                return {
                    "success": True,
                    "original_amount": original_amount,
                    "counterfactual_amount": float(current_amount),
                    "message": (
                        f"If the amount were ${current_amount:,.2f} instead of "
                        f"${original_amount:,.2f}, this would not be flagged."
                    ),
                }

        return {
            "success": False,
            "original_amount": original_amount,
            "message": "No nearby counterfactual found by amount reduction alone.",
        }

    def extract_rules_from_isolation_forest(
        self,
        detector: Any,
        sample: Any,
        feature_names: Optional[List[str]] = None,
        max_trees: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract human-readable rules from Isolation Forest tree paths.
        """
        if not hasattr(detector, "model"):
            return []

        model = detector.model
        sample = np.asarray(sample)

        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        rules = []

        for tree_idx, estimator in enumerate(model.estimators_[:max_trees]):
            tree = estimator.tree_
            node_indicator = estimator.decision_path(sample)
            node_indices = node_indicator.indices

            path_rules = []

            for node_id in node_indices:
                if tree.feature[node_id] != -2:
                    feature_idx = tree.feature[node_id]
                    threshold = float(tree.threshold[node_id])
                    value = float(sample[0, feature_idx])

                    feature_name = (
                        feature_names[feature_idx]
                        if feature_names is not None and feature_idx < len(feature_names)
                        else f"feature_{feature_idx}"
                    )

                    if value <= threshold:
                        path_rules.append(
                            {
                                "tree": tree_idx,
                                "feature": feature_name,
                                "condition": "<=",
                                "threshold": threshold,
                                "sample_value": value,
                            }
                        )
                    else:
                        path_rules.append(
                            {
                                "tree": tree_idx,
                                "feature": feature_name,
                                "condition": ">",
                                "threshold": threshold,
                                "sample_value": value,
                            }
                        )

            rules.extend(path_rules)

        return rules

    def natural_language_summary(
        self,
        explanation: Dict[str, Any],
    ) -> str:
        """
        Converts explanation dict into a human-readable summary.
        """
        reasons = []

        if "top_contributors" in explanation:
            for item in explanation["top_contributors"][:3]:
                reasons.append(
                    f"{item['feature']} is unusual "
                    f"(value={item.get('value', 'N/A')}, contribution={item.get('contribution', 0):.3f})"
                )

        if "top_features" in explanation:
            for item in explanation["top_features"][:3]:
                reasons.append(
                    f"{item['feature']} contributed strongly "
                    f"(value={item.get('sample_value', 'N/A')}, shap={item.get('shap_value', 0):.3f})"
                )

        if "base_detector_scores" in explanation:
            reasons.append(
                "Multiple detectors contributed: "
                + ", ".join(
                    f"{k}={v:.3f}" for k, v in explanation["base_detector_scores"].items()
                )
            )

        if not reasons:
            return "Flagged because the model produced a high anomaly score."

        prefix = "Flagged because:"
        body = "; ".join(f"({i+1}) {r}" for i, r in enumerate(reasons))

        return f"{prefix} {body}"


class ReportGenerator:
    """
    Generates JSON and HTML anomaly reports.
    """

    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(
        self,
        report_id: str,
        payload: Dict[str, Any],
    ) -> Path:
        out_path = self.output_dir / f"{report_id}.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out_path

    def generate_html(
        self,
        report_id: str,
        payload: Dict[str, Any],
    ) -> Path:
        rows = []

        for key, value in payload.items():
            rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Report {report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
    </style>
</head>
<body>
    <h1>Anomaly Report: {report_id}</h1>
    <table>
        <tr><th>Field</th><th>Value</th></tr>
        {''.join(rows)}
    </table>
</body>
</html>
"""

        out_path = self.output_dir / f"{report_id}.html"
        out_path.write_text(html, encoding="utf-8")
        return out_path