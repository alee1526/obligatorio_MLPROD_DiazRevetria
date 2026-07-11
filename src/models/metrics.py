from sklearn.metrics import balanced_accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
