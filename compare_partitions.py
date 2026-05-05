from graphml_to_parquet import convert_graphml_to_parquet
import pandas as pd
import numpy as np
from itertools import combinations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
DATA_DIR = Path("data")

def canonicalize_partition(labels) -> frozenset[frozenset]:
    """
    Convert a partition (Series or list-like) to a label-independent representation:
    a frozenset of frozensets of positional indices.
    """
    s = pd.Series(labels).reset_index(drop=True)
    groups = s.groupby(s).apply(lambda g: frozenset(g.index.tolist()))
    return frozenset(groups)


def jaccard_similarity(set_a: frozenset[frozenset], set_b: frozenset[frozenset]) -> float:
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 1.0


def adjusted_rand_index(labels_a, labels_b) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(labels_a, labels_b)
    except ImportError:
        pass

    n = len(labels_a)
    a = np.array(labels_a)
    b = np.array(labels_b)

    classes_a = np.unique(a)
    classes_b = np.unique(b)

    contingency = np.array([
        [((a == ca) & (b == cb)).sum() for cb in classes_b]
        for ca in classes_a
    ])

    sum_a = contingency.sum(axis=1)
    sum_b = contingency.sum(axis=0)

    def comb2(n):
        return n * (n - 1) / 2

    sum_comb_c = sum(comb2(n_ij) for n_ij in contingency.flatten())
    sum_comb_a = sum(comb2(n_i) for n_i in sum_a)
    sum_comb_b = sum(comb2(n_j) for n_j in sum_b)
    total_comb  = comb2(n)

    expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0
    max_val  = (sum_comb_a + sum_comb_b) / 2
    denom    = max_val - expected

    return (sum_comb_c - expected) / denom if denom != 0 else 1.0


def compare_partitions(
    p1,
    p2=None,
    col_p1: str = "P1",
    col_p2: str = "P2",
    ari_threshold: float = 0.9,
    jaccard_threshold: float = 0.8,
) -> dict:
    """
    Compare two graph partitions.

    Accepted call signatures:
        compare_partitions(df)                        # DataFrame + default column names
        compare_partitions(df, col_p1="A", col_p2="B")  # DataFrame + custom column names
        compare_partitions(series_a, series_b)        # two Series (or any list-like)
    """
    if isinstance(p1, pd.DataFrame):
        labels_a = p1[col_p1].tolist()
        labels_b = p1[col_p2].tolist()
    elif p2 is not None:
        labels_a = p1
        labels_b = p2
    else:
        raise ValueError(
            "Pass either a single DataFrame with col_p1/col_p2, "
            "or two Series / list-like objects."
        )

    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Partition lengths differ: {len(labels_a)} vs {len(labels_b)}."
        )
    print(f"Partition in labels_a has {len(labels_a.unique())} groups, whereas partition in labels_b has {len(labels_b.unique())} groups!")
    canon_p1 = canonicalize_partition(labels_a)
    canon_p2 = canonicalize_partition(labels_b)

    identical = canon_p1 == canon_p2
    ari       = adjusted_rand_index(labels_a, labels_b)
    jaccard   = jaccard_similarity(canon_p1, canon_p2)

    if identical:
        verdict = "✅ Partitions are structurally identical (labels differ, groupings match)."
    elif ari >= ari_threshold and jaccard >= jaccard_threshold:
        verdict = f"🟡 Partitions are very similar (ARI={ari:.3f}, Jaccard={jaccard:.3f})."
    elif ari > 0.5:
        verdict = f"🟠 Partitions are moderately similar (ARI={ari:.3f}, Jaccard={jaccard:.3f})."
    else:
        verdict = f"🔴 Partitions differ significantly (ARI={ari:.3f}, Jaccard={jaccard:.3f})."

    return {
        "identical":   identical,
        "ari":         round(ari, 4),
        "jaccard":     round(jaccard, 4),
        "n_groups_p1": len(canon_p1),
        "n_groups_p2": len(canon_p2),
        "verdict":     verdict,
    }

df = convert_graphml_to_parquet(DATA_DIR / "citation_network_with_topics.graphml", save_to_disk=False)
df_new = convert_graphml_to_parquet(DATA_DIR / "citation_network_with_topics_new.graphml", save_to_disk=False)
result = compare_partitions(df_new["topic"], df["topic"])
for key, val in result.items():
    print(f"{key:12s}: {val}")
    
# df_low_res= pd.read_parquet(DATA_DIR / "clean_unified_database_with_communities_low_res.parquet")
# result = compare_partitions(df_low_res["cpm_communities_at_res=0.003"], df_low_res["Cluster"])
# for key, val in result.items():
#     print(f"{key:12s}: {val}")

# df_low_res_before_gephi = pd.read_parquet(DATA_DIR / "clean_unified_database_with_communities_low_res_before_gephi.parquet")
# df_mid_res_before_gephi = pd.read_parquet(DATA_DIR / "clean_unified_database_with_communities_mid_res_before_gephi.parquet")
# result = compare_partitions(df_low_res_before_gephi["cpm_communities_at_res=0.003"], df_low_res_before_gephi["cpm_communities_at_res=0.004"])
# for key, val in result.items():
    # print(f"{key:12s}: {val}")