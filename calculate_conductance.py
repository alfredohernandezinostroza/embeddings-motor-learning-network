"""
Conductance per community calculator for Leiden partitions stored in GraphML files.

For each 'cpm_communities_at_res=X' attribute found in the graph, computes:
  - Per-community conductance
  - Mean, median, and worst-case (max) conductance across communities
  - A null baseline by randomly reshuffling the partition

Usage:
    python conductance_per_community.py <path_to_graphml> [--null-runs 100]
"""

import argparse
import random
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def conductance(G: nx.Graph, community_nodes: set) -> float:
    r"""
    Conductance of a single community S in graph G.

        phi(S) = cut(S, V\S) / min(vol(S), vol(V\S))

    Returns NaN if the community is empty or covers the whole graph,
    and 0.0 if vol(S) == 0 (isolated nodes only).
    """
    S = set(community_nodes)
    complement = set(G.nodes()) - S

    if not S or not complement:
        return float("nan")

    # Number of edges crossing the cut (for directed graphs, count both directions)
    if G.is_directed():
        cut = sum(
            1 for u in S for v in G.successors(u) if v in complement
        ) + sum(
            1 for u in S for v in G.predecessors(u) if v in complement
        )
        vol_S = sum(G.in_degree(n) + G.out_degree(n) for n in S)
        vol_comp = sum(G.in_degree(n) + G.out_degree(n) for n in complement)
    else:
        cut = nx.cut_size(G, S)
        vol_S = sum(d for _, d in G.degree(S))
        vol_comp = sum(d for _, d in G.degree(complement))

    denom = min(vol_S, vol_comp)
    if denom == 0:
        return 0.0

    return cut / denom


# ---------------------------------------------------------------------------
# Per-resolution analysis
# ---------------------------------------------------------------------------

def compute_conductances_for_partition(G: nx.Graph, partition_attr: str) -> pd.DataFrame:
    """
    Given a node attribute that maps each node to its community ID,
    compute conductance for every community.

    Returns a DataFrame with columns: community, size, conductance
    """
    # Group nodes by community
    communities = defaultdict(set)
    for node, data in G.nodes(data=True):
        comm = data.get(partition_attr)
        if comm is not None:
            communities[comm].add(node)

    records = []
    for comm_id, nodes in communities.items():
        phi = conductance(G, nodes)
        records.append({
            "community": comm_id,
            "size": len(nodes),
            "conductance": phi,
        })

    df = pd.DataFrame(records).sort_values("community").reset_index(drop=True)
    return df


def summarise(df: pd.DataFrame) -> dict:
    valid = df["conductance"].dropna()
    return {
        "n_communities": len(df),
        "mean_conductance": valid.mean(),
        "median_conductance": valid.median(),
        "max_conductance": valid.max(),   # worst community
        "min_conductance": valid.min(),   # best community
        "std_conductance": valid.std(),
    }


# ---------------------------------------------------------------------------
# Null baseline
# ---------------------------------------------------------------------------

def null_baseline_conductance(
    G: nx.Graph, partition_attr: str, n_runs: int = 100
) -> dict:
    """
    Randomly shuffle the community labels (preserving community sizes)
    and compute mean conductance. Repeat n_runs times.

    Returns mean and std of the null mean-conductance.
    """
    # Get original community sizes
    communities = defaultdict(set)
    for node, data in G.nodes(data=True):
        comm = data.get(partition_attr)
        if comm is not None:
            communities[comm].add(node)

    sizes = [len(v) for v in communities.values()]
    nodes = list(G.nodes())

    null_means = []
    for _ in range(n_runs):
        random.shuffle(nodes)
        idx = 0
        null_phi = []
        for size in sizes:
            shuffled_comm = set(nodes[idx: idx + size])
            idx += size
            phi = conductance(G, shuffled_comm)
            if not np.isnan(phi):
                null_phi.append(phi)
        if null_phi:
            null_means.append(np.mean(null_phi))

    return {
        "null_mean_conductance_mean": np.mean(null_means),
        "null_mean_conductance_std": np.std(null_means),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_partition_attributes(G: nx.Graph) -> list[str]:
    """Return all node attributes that look like Leiden CPM community labels."""
    sample_node = next(iter(G.nodes(data=True)))[1]
    return sorted(
        attr for attr in sample_node
        # if attr.startswith("cpm_communities_at_res=")
        if attr.startswith("Cluster")
    )


def main(graphml_path: str, null_runs: int = 100):
    print(f"Loading graph from: {graphml_path}")
    G = nx.read_graphml(graphml_path, )
    G.to_undirected() 
    print(f"  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
    print(f"  Directed: {G.is_directed()}\n")

    partition_attrs = find_partition_attributes(G)
    if not partition_attrs:
        raise ValueError(
            "No 'cpm_communities_at_res=X' attributes found in the GraphML file."
        )

    print(f"Found {len(partition_attrs)} resolution(s): "
          f"{[a.split('=')[1] if '=' in a else a for a in partition_attrs]}\n")

    all_summaries = []

    for attr in partition_attrs:
        resolution = attr.split("=")[1] if '=' in attr else attr
        print(f"{'='*60}")
        print(f"Resolution: {resolution}  (attribute: '{attr}')")
        print(f"{'='*60}")

        df = compute_conductances_for_partition(G, attr)
        summary = summarise(df)

        print(f"  Communities       : {summary['n_communities']}")
        print(f"  Mean conductance  : {summary['mean_conductance']:.4f}")
        print(f"  Median conductance: {summary['median_conductance']:.4f}")
        print(f"  Best  community φ : {summary['min_conductance']:.4f}")
        print(f"  Worst community φ : {summary['max_conductance']:.4f}")

        if null_runs > 0:
            null = null_baseline_conductance(G, attr, n_runs=null_runs)
            print(f"  Null baseline φ   : {null['null_mean_conductance_mean']:.4f} "
                  f"± {null['null_mean_conductance_std']:.4f}  ({null_runs} shuffles)")
            improvement = (
                (null['null_mean_conductance_mean'] - summary['mean_conductance'])
                / null['null_mean_conductance_mean'] * 100
            )
            print(f"  Improvement vs null: {improvement:.1f}%")
            summary.update(null)

        summary["resolution"] = resolution
        all_summaries.append(summary)

        # Per-community table (first 20 in order for brevity)
        print(f"\n  Per-community conductance first 20):")
        print(
            df
            .head(20)
            .to_string(index=False, float_format="{:.4f}".format)
        )
        print()

        # Per-community table (top 20 by conductance for brevity)
        print(f"\n  Per-community conductance (sorted by conductance, top 20):")
        print(
            df.sort_values("conductance")
            .head(20)
            .to_string(index=False, float_format="{:.4f}".format)
        )
        print()

        # Save per-community CSV
        out_csv = graphml_path.replace(".graphml", f"_conductance_res{resolution}_with_null_runs_{null_runs}.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}\n")

        # Summary across all resolutions
        summary_df = pd.DataFrame(all_summaries).set_index("resolution")
        print(f"\n{'='*60}")
        print("Summary across all resolutions:")

        # Select columns based on whether null runs were performed
        base_cols = ["n_communities", "mean_conductance", "median_conductance", "max_conductance"]
        if null_runs > 0:
            display_cols = base_cols + ["null_mean_conductance_mean"]
        else:
            display_cols = base_cols

        print(summary_df[display_cols].to_string(float_format="{:.4f}".format))

        summary_csv = graphml_path.replace(".graphml", f"_conductance_summary_with_null_runs_{null_runs}.csv")
        summary_df.to_csv(summary_csv)
        print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute conductance per Leiden community in a GraphML file."
    )
    parser.add_argument("graphml", help="Path to the .graphml file")
    parser.add_argument(
        "--null-runs",
        type=int,
        default=100,
        help="Number of random shuffles for null baseline (0 to skip, default: 100)",
    )
    args = parser.parse_args()
    main(args.graphml, null_runs=args.null_runs)