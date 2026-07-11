import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset_utils import TARGET

GROUP = "patient_id"


def make_split(df, n_splits=7, seed=42, group=GROUP, target=TARGET):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for i, (_, idx) in enumerate(skf.split(df, df[target], groups=df[group])):
        fold.iloc[idx] = i
    names = {0: "test", 1: "val"}
    out = df.copy()
    out["split"] = [names.get(f, "train") for f in fold]
    return out


def patient_overlap(df, group=GROUP):
    per_group = df.groupby(group)["split"].nunique()
    return per_group[per_group > 1]


def class_distribution(df, target=TARGET):
    counts = pd.crosstab(df[target], df["split"])
    props = pd.crosstab(df[target], df["split"], normalize="columns").round(3)
    return counts, props
