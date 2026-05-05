"""
Topic Modeling with BERTopic and Specter2
==========================================
Loads the citation network graph, extracts paper titles and abstracts,
computes Specter2 embeddings, applies BERTopic to discover topics,
and adds the topic assignments as a new node attribute.

Outputs: Updated GraphML file with 'topic' attribute for each node.

Caching
-------
Two expensive steps are cached on disk and reused across runs as long as
the source graph has not changed:

  1. Specter2 embeddings  → DATA_DIR/embeddings_cache.npz
  2. BERTopic model       → DATA_DIR/bertopic_model/

A SHA-256 fingerprint is computed from every node's (id, title, abstract)
triple. If the fingerprint matches the one stored in the cache, the cached
artefacts are loaded instead of recomputed. Any change to the graph (new
nodes, edited abstracts, etc.) automatically busts both caches.

Pass --recompute on the command line to force a full rerun regardless.
"""

import os
import sys
import hashlib
import json
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

DATA_DIR = Path("data")
GRAPHML_FILE = DATA_DIR / "citation_network_selected.graphml"
SUFFIX = "_new"

# ── Configuration ──────────────────────────────────────────────────────────
SPECTER2_MODEL   = "allenai/specter2_base"
OUTPUT_GRAPHML   = DATA_DIR / f"citation_network_with_topics{SUFFIX}.graphml"
EMBED_CACHE      = DATA_DIR / "embeddings_cache.npz"
TOPIC_MODEL_DIR  = DATA_DIR / f"bertopic_model{SUFFIX}_results"
FINGERPRINT_FILE = DATA_DIR / "graph_fingerprint.json"
MIN_TOPIC_SIZE   = 15
N_NEIGHBORS      = 15
N_COMPONENTS     = 5
BATCH_SIZE       = 32

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--recompute", action="store_true",
    help="Ignore all caches and recompute everything from scratch.",
)
args = parser.parse_args()

print("=" * 80)
print("Topic Modeling with BERTopic and Specter2")
print("=" * 80)

# ── Helpers ────────────────────────────────────────────────────────────────

def compute_graph_fingerprint(graph: nx.Graph) -> str:
    """
    SHA-256 over every node's (id, title, abstract) sorted by node id.
    Changes whenever nodes are added/removed or any text field is edited.
    """
    h = hashlib.sha256()
    for node in sorted(graph.nodes()):
        data  = graph.nodes[node]
        entry = f"{node}|{data.get('title','').strip()}|{data.get('abstract','').strip()}"
        h.update(entry.encode())
    return h.hexdigest()


def load_fingerprint() -> str | None:
    if FINGERPRINT_FILE.exists():
        return json.loads(FINGERPRINT_FILE.read_text()).get("fingerprint")
    return None


def save_fingerprint(fp: str) -> None:
    FINGERPRINT_FILE.write_text(json.dumps({"fingerprint": fp}))


# ── 1. Load Graph ─────────────────────────────────────────────────────────
print("\n[1/6] Loading graph...")
G = nx.read_graphml(GRAPHML_FILE)
print(f"  Loaded {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")

# ── 2. Extract Text Data ──────────────────────────────────────────────────
print("\n[2/6] Extracting titles and abstracts...")
documents    = []
node_ids     = []
valid_count  = 0
missing_text = 0

for node, data in G.nodes(data=True):
    title    = data.get('title', '').strip()
    abstract = data.get('abstract', '').strip()
    if title or abstract:
        text = f"{title} [SEP] {abstract}" if abstract else title
        documents.append(text)
        node_ids.append(node)
        valid_count += 1
    else:
        missing_text += 1
        documents.append("")
        node_ids.append(node)

print(f"  Papers with text: {valid_count:,}")
print(f"  Papers without text: {missing_text:,}")

valid_docs    = [doc for doc in documents if doc]
valid_indices = [i for i, doc in enumerate(documents) if doc]

# ── Cache invalidation check ──────────────────────────────────────────────
current_fp  = compute_graph_fingerprint(G)
cached_fp   = None if args.recompute else load_fingerprint()
cache_valid = (current_fp == cached_fp)

if args.recompute:
    print("\n  --recompute flag set: ignoring all caches.")
elif cache_valid:
    print("\n  Graph fingerprint matches cache — cached artefacts will be reused.")
else:
    print("\n  Graph fingerprint changed (or no cache found) — will recompute.")

# ── 3. Specter2 Embeddings (cached) ───────────────────────────────────────
print("\n[3/6] Computing Specter2 embeddings...")

embed_cache_hit = cache_valid and EMBED_CACHE.exists() and not args.recompute

if embed_cache_hit:
    print(f"  Cache hit — loading embeddings from {EMBED_CACHE}")
    valid_embeddings = np.load(EMBED_CACHE)["valid_embeddings"]
    print(f"  Loaded embeddings: {valid_embeddings.shape}")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading model: {SPECTER2_MODEL}  (device: {device})")

    tokenizer = AutoTokenizer.from_pretrained(SPECTER2_MODEL)
    model     = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
    model.load_adapter(
        "allenai/specter2",
        source="hf",
        load_as="specter2",
        set_active=True,
    )
    model = model.to(device)
    model.eval()

    def embed_batch(texts):
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    print(f"  Embedding {len(valid_docs):,} documents in batches of {BATCH_SIZE}...")
    embeddings_list = []
    for i in tqdm(range(0, len(valid_docs), BATCH_SIZE), desc="  Embedding"):
        embeddings_list.append(embed_batch(valid_docs[i:i + BATCH_SIZE]))

    valid_embeddings = np.vstack(embeddings_list)
    print(f"  Embedding shape: {valid_embeddings.shape}")

    np.savez_compressed(EMBED_CACHE, valid_embeddings=valid_embeddings)
    print(f"  Embeddings saved to cache: {EMBED_CACHE}")

    # Free GPU memory — the model is no longer needed past this point
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Reconstruct full array (zeros for nodes without text)
embeddings = np.zeros((len(documents), valid_embeddings.shape[1]))
embeddings[valid_indices] = valid_embeddings

# ── 4. BERTopic (cached) ──────────────────────────────────────────────────
print("\n[4/6] Applying BERTopic clustering...")

topic_cache_hit = (
    cache_valid
    and TOPIC_MODEL_DIR.exists()
    and (DATA_DIR / f"document_topics{SUFFIX}.csv").exists()
    and not args.recompute
)

if topic_cache_hit:
    print(f"  Cache hit — loading BERTopic model from {TOPIC_MODEL_DIR}")
    topic_model = BERTopic.load(str(TOPIC_MODEL_DIR))
    topics      = pd.read_csv(DATA_DIR / f"document_topics{SUFFIX}")["topic"].to_numpy()
    topic_info  = topic_model.get_topic_info()
    print(f"  Loaded model with {len(topic_info) - 1} topics")
else:
    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
        min_dist=0.0, metric='cosine', random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE, metric='euclidean',
        cluster_selection_method='eom', prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words='english', max_features=10000, ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model, top_n_words=10,
        verbose=True, calculate_probabilities=False,
    )

    print("  Fitting BERTopic model...")
    topics_valid, _ = topic_model.fit_transform(valid_docs, valid_embeddings)

    vec = topic_model.vectorizer_model
    if hasattr(vec, "vocabulary_") and vec.vocabulary_ is not None:
        vec.vocabulary_ = {k: int(v) for k, v in vec.vocabulary_.items()}

    topics = np.full(len(documents), -1, dtype=int)
    topics[valid_indices] = topics_valid
    topic_info = topic_model.get_topic_info()

    TOPIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(
        str(TOPIC_MODEL_DIR), serialization="pytorch", save_ctfidf=True,
    )
    print(f"  BERTopic model saved to {TOPIC_MODEL_DIR/"bertopic_model"}")

# Persist fingerprint only after all heavy work succeeds
save_fingerprint(current_fp)

# ── 5. Add Topics to Graph ────────────────────────────────────────────────
print("\n[5/6] Adding topics to graph nodes...")
for i, node in enumerate(node_ids):
    G.nodes[node]['topic'] = int(topics[i])

unique_topics = np.unique(topics)
n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
outliers = np.sum(topics == -1)

print(f"  Number of topics found: {n_topics}")
print(f"  Outliers (topic -1): {outliers:,} ({100*outliers/len(topics):.1f}%)")

topic_counts = pd.Series(topics).value_counts().sort_index()
print(f"\n  Topic size distribution:")
print(f"    Mean:   {topic_counts[topic_counts.index != -1].mean():.1f}")
print(f"    Median: {topic_counts[topic_counts.index != -1].median():.1f}")
print(f"    Max:    {topic_counts[topic_counts.index != -1].max()}")

print(f"\n  Top 10 largest topics:")
for topic_id in topic_counts[topic_counts.index != -1].head(10).index:
    words = topic_model.get_topic(int(topic_id))
    if words:
        top_words = ", ".join(w for w, _ in words[:5])
        print(f"    Topic {topic_id:3d}: {topic_counts[topic_id]:5,} papers — {top_words}")

# ── 6. Save Updated Graph ─────────────────────────────────────────────────
print(f"\n[6/6] Saving updated graph to {OUTPUT_GRAPHML}...")
nx.write_graphml(G, OUTPUT_GRAPHML)
print("  Saved successfully!")

# ── 7. Save Topic Artefacts ───────────────────────────────────────────────
print("\n[Bonus] Saving topic information...")

topic_info.to_csv(DATA_DIR / f"topic_info{SUFFIX}.csv", index=False)
print(f"  Saved topic_info{SUFFIX}.csv")

pd.DataFrame({
    'node_id':  node_ids,
    'topic':    topics,
    'document': documents,
}).to_csv(DATA_DIR / f"document_topics{SUFFIX}.csv", index=False)
print(f"  Saved document_topics{SUFFIX}.csv")

valid_topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
topic_words = []
for topic_id in valid_topic_ids:
    words = topic_model.get_topic(int(topic_id))
    if words:
        topic_words.append({
            'topic_id': topic_id,
            'words':    " | ".join(w for w, _ in words),
            'scores':   " | ".join(f"{s:.4f}" for _, s in words),
        })

pd.DataFrame(topic_words).to_csv(DATA_DIR / f"topic_words{SUFFIX}.csv", index=False)
print(f"  Saved topic_words{SUFFIX}.csv")

print("\n" + "=" * 80)
print("Topic modeling complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Processed {len(documents):,} papers")
print(f"  - Found {n_topics} topics")
print(f"  - Updated graph saved to:          {OUTPUT_GRAPHML}")
print(f"  - Topic info saved to:             {DATA_DIR / f'topic_info{SUFFIX}.csv'}")
print(f"  - Document-topic mapping saved to: {DATA_DIR / f'document_topics{SUFFIX}.csv'}")
print(f"  - Topic words saved to:            {DATA_DIR / f'topic_words{SUFFIX}.csv'}")
print(f"  - Embedding cache:                 {EMBED_CACHE}")
print(f"  - BERTopic model cache:            {TOPIC_MODEL_DIR}")