from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from schemas import ModelMetrics

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Evaluation suite specialized for rare-event financial anomaly detection.
    """

    def __init__(self, top_k: int = 100):
        self.top_k = top_k

    def precision_at_top_k(
        self,
        y_true: Sequence[int],
        y_scores: Sequence[float],
        k: int,
    ) -> float:
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        if len(y_true) == 0 or k <= 0:
            return 0.0

        k = min(k, len(y_true))
        order = np.argsort(y_scores)[::-1]
        top_k_true = y_true[order[:k]]

        return float(np.mean(top_k_true))

    def compute_metrics(
        self,
        model_name: str,
        y_true: Sequence[int],
        y_pred_scores: Sequence[float],
        y_pred_labels: Sequence[int],
        k: Optional[int] = None,
    ) -> ModelMetrics:
        y_true = np.asarray(y_true).astype(int)
        y_pred_scores = np.asarray(y_pred_scores).astype(float)
        y_pred_labels = np.asarray(y_pred_labels).astype(int)

        k = k or self.top_k

        metrics = ModelMetrics(model_name=model_name)

        if len(y_true) == 0:
            return metrics

        metrics.accuracy = float(accuracy_score(y_true, y_pred_labels))
        metrics.precision = float(precision_score(y_true, y_pred_labels, zero_division=0))
        metrics.recall = float(recall_score(y_true, y_pred_labels, zero_division=0))
        metrics.f1 = float(f1_score(y_true, y_pred_labels, zero_division=0))

        try:
            metrics.roc_auc = float(roc_auc_score(y_true, y_pred_scores))
        except ValueError:
            metrics.roc_auc = 0.0

        try:
            precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_scores)
            metrics.pr_auc = float(auc(recall_arr, precision_arr))
        except ValueError:
            metrics.pr_auc = 0.0

        metrics.precision_at_k = self.precision_at_top_k(y_true, y_pred_scores, k)
        metrics.alert_fatigue_score = metrics.precision_at_k

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred_labels, labels=[0, 1]
        ).ravel()

        metrics.false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        metrics.extra.update(
            {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "top_k": int(k),
            }
        )

        return metrics

    def plot_precision_recall_curve(
        self,
        y_true: Sequence[int],
        y_scores: Sequence[float],
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_arr, precision_arr, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def plot_roc_curve(
        self,
        y_true: Sequence[int],
        y_scores: Sequence[float],
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def compute_confusion_matrix_cost_sensitive(
        self,
        y_true: Sequence[int],
        y_pred_labels: Sequence[int],
        cost_matrix: Dict[str, float],
    ) -> float:
        """
        cost_matrix keys:
        - fp_cost
        - fn_cost
        - tp_cost
        - tn_cost
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred_labels = np.asarray(y_pred_labels).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred_labels, labels=[0, 1]
        ).ravel()

        cost = (
            fp * cost_matrix.get("fp_cost", 1.0)
            + fn * cost_matrix.get("fn_cost", 10.0)
            + tp * cost_matrix.get("tp_cost", 0.0)
            + tn * cost_matrix.get("tn_cost", 0.0)
        )

        return float(cost)

    def detection_delay_analysis(
        self,
        true_events: pd.DataFrame,
        alerts: pd.DataFrame,
        event_time_col: str = "timestamp",
        alert_time_col: str = "timestamp",
        entity_col: str = "user_id",
        freq: str = "h",
    ) -> Dict[str, Any]:
        """
        Measures how long after a true event the first alert occurs.

        true_events columns:
        - entity_col
        - event_time_col

        alerts columns:
        - entity_col
        - alert_time_col
        """
        true_events = true_events.copy()
        alerts = alerts.copy()

        true_events[event_time_col] = pd.to_datetime(true_events[event_time_col], utc=True)
        alerts[alert_time_col] = pd.to_datetime(alerts[alert_time_col], utc=True)

        true_events = true_events.sort_values(event_time_col)
        alerts = alerts.sort_values(alert_time_col)

        merged = pd.merge_asof(
            true_events,
            alerts.rename(columns={alert_time_col: "matched_alert_time"}),
            left_on=event_time_col,
            right_on="matched_alert_time",
            by=entity_col,
            direction="forward",
        )

        merged["delay"] = merged["matched_alert_time"] - merged[event_time_col]

        detected = merged.dropna(subset=["matched_alert_time"])

        if len(detected) == 0:
            return {
                "avg_delay_periods": None,
                "median_delay_periods": None,
                "detection_rate": 0.0,
            }

        if freq == "h":
            delays = detected["delay"].dt.total_seconds() / 3600.0
        elif freq == "D":
            delays = detected["delay"].dt.total_seconds() / 86400.0
        else:
            delays = detected["delay"].dt.total_seconds()

        return {
            "avg_delay_periods": float(delays.mean()),
            "median_delay_periods": float(delays.median()),
            "detection_rate": float(len(detected) / len(true_events)),
        }

    def false_positive_breakdown(
        self,
        df: pd.DataFrame,
        y_true: Sequence[int],
        y_pred_labels: Sequence[int],
        breakdown_cols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyzes false positives by merchant category, time of day, user segment, etc.
        """
        tmp = df.copy()
        tmp["y_true"] = np.asarray(y_true).astype(int)
        tmp["y_pred"] = np.asarray(y_pred_labels).astype(int)

        fp_mask = (tmp["y_true"] == 0) & (tmp["y_pred"] == 1)
        fp_df = tmp.loc[fp_mask]

        result = {}

        for col in breakdown_cols:
            if col not in fp_df.columns:
                continue

            result[col] = (
                fp_df.groupby(col)
                .size()
                .reset_index(name="false_positive_count")
                .sort_values("false_positive_count", ascending=False)
            )

        return result

    def var_reduction(
        self,
        losses: Sequence[float],
        detected_flags: Sequence[bool],
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Approximate Value-at-Risk reduction from early detection.

        Idea:
        - VaR before detection uses full loss distribution
        - VaR after detection removes detected losses
        - reduction includes avoided tail loss plus VaR improvement
        """
        losses = np.asarray(losses, dtype=float)
        detected_flags = np.asarray(detected_flags).astype(bool)

        if len(losses) == 0:
            return {
                "var_before": 0.0,
                "var_after": 0.0,
                "var_reduction": 0.0,
                "avoided_loss": 0.0,
            }

        var_before = float(np.quantile(losses, confidence))

        remaining_losses = losses[~detected_flags]
        var_after = float(np.quantile(remaining_losses, confidence)) if len(remaining_losses) > 0 else 0.0

        avoided_loss = float(np.sum(losses[detected_flags]))

        return {
            "var_before": var_before,
            "var_after": var_after,
            "var_reduction": max(0.0, var_before - var_after),
            "avoided_loss": avoided_loss,
        }

    def cascade_detection_rate(
        self,
        detected_ids: Sequence[str],
        related_groups: List[set],
    ) -> float:
        """
        Measures whether detecting one anomaly helps discover related anomalies.

        related_groups:
        - list of sets of related anomaly IDs
        """
        if not related_groups:
            return 0.0

        detected_set = set(detected_ids)

        hits = 0
        for group in related_groups:
            if len(group & detected_set) > 0:
                hits += 1

        return float(hits / len(related_groups))