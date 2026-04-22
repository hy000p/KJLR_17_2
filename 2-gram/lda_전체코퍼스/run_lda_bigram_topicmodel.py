# -*- coding: utf-8 -*-
"""
LDA topic modeling on 2-grams (character bigrams) from Joseon-era Gyehoesi poems.
- Source: first sheet of "조선시대 계회시 목록.xlsx" (sheet: 계회시_저자)
- 2-grams are extracted ONLY within (a) punctuation-delimited lines/phrases and (b) meter-based chunks (4/5/6/7 chars).
- Topic number K is selected by: (log-odds z for topic-distinctive terms) + (NPMI coherence of those terms).
Outputs (PNG + HTML) are written to ./lda_bigram_out/

Tested with: Python 3.11, scikit-learn, scipy, matplotlib, pandas, pyLDAvis.
"""

import os, re, math, warnings
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")

# -----------------------------
# 0) Paths
# -----------------------------
EXCEL_PATH = "조선시대 계회시 목록.xlsx"   # <-- adjust if needed
SHEET_INDEX = 0                           # first sheet
OUT_DIR = "lda_bigram_out"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Preprocess: within-line + meter-based chunking
# -----------------------------
STOP_CHARS = set("之 其 而 以 於 于 也 矣 焉 兮 乎 哉 者 所 乃 與 及 且 為 曰 云 則 非 無 有 不 復 亦 又 更 可 何 我 吾 爾 汝 彼 此 是 斯 相 諸 各 皆 每".split())
NUMERALS = set("一 二 三 四 五 六 七 八 九 十 百 千 萬 兩 雙 第".split())
STOP_CHARS = STOP_CHARS - NUMERALS  # keep numerals

KEEP_RE = re.compile(r'[\u4e00-\u9fff]')  # CJK Unified Ideographs
SPLIT_RE = re.compile(r'[。！？；，、,;:\n\r\t/／\|]+')

def infer_meter(val):
    """Return meter length (4/5/6/7) or None."""
    if pd.isna(val):
        return None
    s = str(val)
    nums = sorted(set(re.findall(r'[4567]', s)))
    if len(nums) == 1:
        return int(nums[0])
    return None

def clean_segment(seg: str) -> str:
    """Keep only CJK characters."""
    return ''.join(ch for ch in seg if KEEP_RE.match(ch))

def chunk_by_meter(seg: str, m: int | None):
    """Split seg into fixed-length chunks (m chars)."""
    if m is None:
        return [seg] if seg else []
    if len(seg) <= m:
        return [seg] if seg else []
    return [seg[i:i+m] for i in range(0, len(seg), m) if seg[i:i+m]]

def extract_bigrams(text: str, meter: int | None):
    """Extract character 2-grams; do not cross line/phrase boundaries."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p for p in SPLIT_RE.split(text) if p]
    toks = []
    for p in parts:
        p = clean_segment(p)
        if not p:
            continue
        for chunk in chunk_by_meter(p, meter):
            if len(chunk) < 2:
                continue
            for i in range(len(chunk) - 1):
                bg = chunk[i:i+2]
                # conservative stop-filter: drop if either char is a function word (numerals kept)
                if (bg[0] in STOP_CHARS) or (bg[1] in STOP_CHARS):
                    continue
                toks.append(bg)
    return toks

# -----------------------------
# 2) Load data & build corpus
# -----------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_INDEX)

meters = df["문체"].apply(infer_meter)
token_lists = [extract_bigrams(t, m) for t, m in zip(df["원문"], meters)]

docs = [" ".join(toks) for toks in token_lists]
mask = np.array([len(toks) > 0 for toks in token_lists])
docs = [d for d, ok in zip(docs, mask) if ok]
df = df.loc[mask].reset_index(drop=True)

print(f"[INFO] docs={len(docs)} (non-empty), raw rows={mask.size}")

# -----------------------------
# 3) Vectorize
# -----------------------------
vectorizer = CountVectorizer(
    tokenizer=str.split,
    preprocessor=lambda x: x,
    token_pattern=None,
    min_df=2,
    max_df=0.7,
    max_features=8000
)
X = vectorizer.fit_transform(docs)  # (D,V)
vocab = vectorizer.get_feature_names_out()
term_freq = np.asarray(X.sum(axis=0)).ravel()
doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
X_bin = X.copy().tocsr()
X_bin.data = np.ones_like(X_bin.data)

print(f"[INFO] DTM shape={X.shape}, vocab={len(vocab)}, tokens={int(X.sum())}")

# -----------------------------
# 4) log-odds z + NPMI coherence
# -----------------------------
alpha_w = term_freq + 0.01  # informative Dirichlet prior (Monroe et al.)

def topic_logodds_z(y_i, y_j, alpha_w):
    """Monroe-style log-odds z comparing group i vs j."""
    n_i = y_i.sum()
    n_j = y_j.sum()
    alpha0 = alpha_w.sum()

    numer_i = y_i + alpha_w
    denom_i = (n_i + alpha0) - numer_i
    numer_j = y_j + alpha_w
    denom_j = (n_j + alpha0) - numer_j

    denom_i = np.clip(denom_i, 1e-12, None)
    denom_j = np.clip(denom_j, 1e-12, None)

    delta = np.log(numer_i / denom_i) - np.log(numer_j / denom_j)
    var = 1.0 / numer_i + 1.0 / numer_j
    return delta / np.sqrt(var)

def npmi(p12, p1, p2):
    if p12 <= 0 or p1 <= 0 or p2 <= 0:
        return -1.0
    pmi = math.log(p12 / (p1 * p2))
    return pmi / (-math.log(p12))

def coherence_npmi_for_terms(term_indices):
    """Average pairwise NPMI using document co-occurrence."""
    idxs = list(term_indices)
    if len(idxs) < 2:
        return float("nan")
    N_docs = X.shape[0]
    p = doc_freq[idxs] / N_docs
    s = 0.0
    cnt = 0
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            i, j = idxs[a], idxs[b]
            df12 = float(X_bin[:, i].multiply(X_bin[:, j]).sum())
            p12 = df12 / N_docs
            s += npmi(p12, p[a], p[b])
            cnt += 1
    return s / cnt if cnt else float("nan")

def avg_jaccard(list_of_sets):
    js = []
    for i in range(len(list_of_sets)):
        for j in range(i + 1, len(list_of_sets)):
            a, b = list_of_sets[i], list_of_sets[j]
            inter = len(a & b)
            union = len(a | b)
            js.append(inter / union if union else 0.0)
    return float(np.mean(js)) if js else 0.0

def fit_lda(K, max_iter=40, seed=42):
    lda = LatentDirichletAllocation(
        n_components=K,
        learning_method="batch",
        max_iter=max_iter,
        random_state=seed,
        evaluate_every=0,
        n_jobs=1,
        max_doc_update_iter=50
    )
    lda.fit(X)
    return lda

def eval_K(lda, topn=15):
    """Compute mean coherence over topics using topn terms selected by log-odds z."""
    theta = lda.transform(X)
    doc_lengths = np.asarray(X.sum(axis=1)).ravel()
    topic_sizes = (theta * doc_lengths[:, None]).sum(axis=0)
    phi = lda.components_ / lda.components_.sum(axis=1)[:, None]
    topic_counts = phi * topic_sizes[:, None]
    total_counts = topic_counts.sum(axis=0)

    coh = []
    top_sets = []
    for k in range(lda.n_components):
        z = topic_logodds_z(topic_counts[k], total_counts - topic_counts[k], alpha_w)
        top_idx = np.argsort(-z)[:topn]
        top_sets.append(set(top_idx.tolist()))
        coh.append(coherence_npmi_for_terms(top_idx))
    return float(np.nanmean(coh)), avg_jaccard(top_sets)

# -----------------------------
# 5) Choose K by (log-odds z + NPMI)
# -----------------------------
K_candidates = list(range(3, 16))
rows = []
models = {}

for K in K_candidates:
    lda_k = fit_lda(K, max_iter=40, seed=42)
    avg_coh, avg_ov = eval_K(lda_k, topn=15)
    rows.append((K, avg_coh, avg_ov))
    models[K] = lda_k
    print(f"[K={K:02d}] coherence={avg_coh: .4f} | overlap={avg_ov: .4f}")

eval_df = pd.DataFrame(rows, columns=["K", "avg_coh", "avg_overlap"])
best_coh = eval_df["avg_coh"].max()
# parsimony rule: choose the smallest K within 0.02 of the best coherence
eligible = eval_df[eval_df["avg_coh"] >= (best_coh - 0.02)].sort_values("K")
K_final = int(eligible.iloc[0]["K"])
print(f"[INFO] Selected K_final={K_final} (best_coh={best_coh:.4f})")

# Refit final model with more iterations for stability
lda = LatentDirichletAllocation(
    n_components=K_final,
    learning_method="batch",
    max_iter=250,
    random_state=42,
    evaluate_every=0,
    n_jobs=1,
    max_doc_update_iter=100
)
lda.fit(X)

# -----------------------------
# 6) Visualizations (PNG) + pyLDAvis (HTML)
# -----------------------------
# Font (for Hanja): fallback to default if not available
plt.rcParams["font.family"] = "NanumMyeongjo"
plt.rcParams["axes.unicode_minus"] = False

# (A) K-selection plot
fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=200)
ax1.plot(eval_df["K"], eval_df["avg_coh"], marker="o")
ax1.axvline(K_final, linestyle="--")
ax1.set_xlabel("Number of topics (K)")
ax1.set_ylabel("Mean NPMI coherence (log-odds-selected terms)")
ax1.set_title("Selecting K using log-odds z + NPMI")
ax2 = ax1.twinx()
ax2.plot(eval_df["K"], eval_df["avg_overlap"], marker="s")
ax2.set_ylabel("Mean Jaccard overlap")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lda_k_selection.png"), bbox_inches="tight")
plt.close(fig)

# Topic-term and doc-topic distributions
phi = (lda.components_ / lda.components_.sum(axis=1)[:, None]).astype(float)
theta = lda.transform(X).astype(float)
doc_lengths = np.asarray(X.sum(axis=1)).ravel().astype(float)
topic_sizes = (theta * doc_lengths[:, None]).sum(axis=0)
topic_shares = topic_sizes / topic_sizes.sum()

# (B) Intertopic distance map (Jensen–Shannon + MDS)
def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)

K = phi.shape[0]
dist = np.zeros((K, K), dtype=float)
for i in range(K):
    for j in range(i + 1, K):
        dist[i, j] = dist[j, i] = js_divergence(phi[i], phi[j])

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
coords = mds.fit_transform(dist)

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
sizes = 2000 * topic_shares
ax.scatter(coords[:, 0], coords[:, 1], s=sizes, alpha=0.6)
for i in range(K):
    ax.text(coords[i, 0], coords[i, 1], str(i + 1), ha="center", va="center", fontsize=9)
ax.set_title("Intertopic distance map (Jensen–Shannon + MDS)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lda_intertopic_map.png"), bbox_inches="tight")
plt.close(fig)

# Helper: top terms by probability
def top_terms_by_prob(k, n=10):
    row = phi[k]
    idx = np.argsort(-row)[:n]
    return vocab[idx], row[idx]

# (C) Topic keywords (probability) as a table-like figure
fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=200)
ax.axis("off")
lines = []
for k in range(K):
    terms, _ = top_terms_by_prob(k, n=10)
    lines.append(f"Topic {k+1:02d} ({topic_shares[k]*100:4.1f}%): " + " · ".join(terms))
ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=10)
ax.set_title("LDA topics (2-gram) — top keywords by probability", pad=20)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lda_topic_keywords.png"), bbox_inches="tight")
plt.close(fig)

# (D) Topic bar charts grid
rows, cols = int(math.ceil(K / 3)), 3
fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), dpi=200)
axes = np.array(axes).reshape(-1)
for k in range(K):
    ax = axes[k]
    terms, probs = top_terms_by_prob(k, n=10)
    terms = terms[::-1]
    probs = probs[::-1]
    ax.barh(range(len(terms)), probs)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_title(f"Topic {k+1:02d} ({topic_shares[k]*100:.1f}%)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
for j in range(K, rows * cols):
    axes[j].axis("off")
fig.suptitle("Top terms per topic (probability)", fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(OUT_DIR, "lda_topic_bars_grid.png"), bbox_inches="tight")
plt.close(fig)

# (E) Distinctive keywords via log-odds z (vs. rest)
def distinctive_terms_by_logodds(k, n=12, min_tf=3):
    # expected token counts per term for each topic
    topic_counts = phi * topic_sizes[:, None]       # (K,V)
    total_counts = topic_counts.sum(axis=0)         # (V,)
    eligible = term_freq >= min_tf
    z = topic_logodds_z(topic_counts[k], total_counts - topic_counts[k], alpha_w)
    z_masked = z.copy()
    z_masked[~eligible] = -np.inf
    idx = np.argsort(-z_masked)[:n]
    return vocab[idx]

fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=200)
ax.axis("off")
lines = []
for k in range(K):
    terms = distinctive_terms_by_logodds(k, n=12, min_tf=3)
    lines.append(f"Topic {k+1:02d} ({topic_shares[k]*100:4.1f}%): " + " · ".join(terms))
ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=10)
ax.set_title("LDA topics — distinctive keywords (log-odds z vs. rest)", pad=20)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lda_topic_distinctive.png"), bbox_inches="tight")
plt.close(fig)

# (F) pyLDAvis (HTML)
# Install if missing:
#   pip install pyldavis==3.4.0
import pyLDAvis

vis = pyLDAvis.prepare(phi, theta, doc_lengths, vocab.tolist(), term_freq.astype(float), sort_topics=False, n_jobs=1)
pyLDAvis.save_html(vis, os.path.join(OUT_DIR, f"pyldavis_lda_bigram_K{K_final}.html"))

print("[DONE] Files written to:", os.path.abspath(OUT_DIR))
