def calculate_metrics(predictions: list[dict], fields: tuple[str, ...]) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_field = {}

    for field in fields:
        tp = fp = fn = 0
        for item in predictions:
            gt = item["ground_truth"][field]
            pred = item["prediction"][field]
            if gt is not None and pred == gt:
                tp += 1
            else:
                if pred is not None:
                    fp += 1
                if gt is not None:
                    fn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_field[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(
                item["ground_truth"][field] is not None for item in predictions
            ),
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    precision = (
        totals["tp"] / (totals["tp"] + totals["fp"])
        if totals["tp"] + totals["fp"]
        else 0.0
    )
    recall = (
        totals["tp"] / (totals["tp"] + totals["fn"])
        if totals["tp"] + totals["fn"]
        else 0.0
    )
    micro_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    document_accuracy = (
        sum(item["document_correct"] for item in predictions) / len(predictions)
        if predictions
        else 0.0
    )

    return {
        "per_field": per_field,
        "document_accuracy": document_accuracy,
        "micro_f1": micro_f1,
    }
