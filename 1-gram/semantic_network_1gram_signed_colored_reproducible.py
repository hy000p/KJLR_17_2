# -*- coding: utf-8 -*-
"""
4관청(one-vs-rest) 차등 네트워크: ONE(해당 관청) vs REST(다른 3관청)
- 엣지 가중치: NPMI
- 차등: d = NPMI(one) - NPMI(rest)
- 시각화: ONE은 컬러, REST는 회색(같은 색), 노드는 빈도 우세 집단으로 색을 따라감
- PNG 저장
"""

import os, re, math
from collections import Counter

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib.patches import Patch

EXCEL_PATH = r"/mnt/data/조선시대 계회시 목록.xlsx"
OUTDIR = r"/mnt/data/1gram_의미연결망_colored_diff"
os.makedirs(OUTDIR, exist_ok=True)

FONT_PATH = r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FP = fm.FontProperties(fname=FONT_PATH)
mpl.rcParams["axes.unicode_minus"] = False

df = pd.read_excel(EXCEL_PATH, sheet_name=0)

stopwords = set(list("之其而以於于也矣焉兮乎哉者所乃與及且為曰云則非無有不復亦又更可何我吾爾汝彼此是斯相諸各皆每"))
numerals  = set(list("一二三四五六七八九十百千萬兩雙第"))
stopwords = stopwords - numerals

aliases = {'사헌부': ['驄馬', '憲府', '栢府', '相臺', '烏臺', '御史臺', '監察司', '大관', '臺官'], '사간원': ['薇院', '諫院', '臺諫', '대간'], '승정원': ['銀臺', '政院', '喉院', '代言司', '대언사'], '의금부': ['金吾', '禁府', '王府', '詔獄', '金部']}
HAN_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
alias_chars=set()
for lst in aliases.values():
    for w in lst:
        alias_chars.update(HAN_RE.findall(str(w)))
stopwords = stopwords - alias_chars

def tokenize_1gram(text):
    if not isinstance(text, str):
        return []
    chars = HAN_RE.findall(text)
    return [c for c in chars if c not in stopwords]

df["tokens"] = df["원문"].apply(tokenize_1gram)

targets = ["사헌부","사간원","승정원","의금부"]
types = df["유형"].astype(str).fillna("").str.strip()
df["office"] = None
for t in targets:
    df.loc[types.str.startswith(t), "office"] = t

def cooc_counts(docs, window=5):
    node = Counter()
    pair = Counter()
    for toks in docs:
        node.update(toks)
        L = len(toks)
        for i in range(L):
            a = toks[i]
            for b in toks[i+1:i+window]:
                if a == b:
                    continue
                x,y = (a,b) if a<b else (b,a)
                pair[(x,y)] += 1
    return node, pair

def edge_npmi(node, pair):
    tot_node = sum(node.values())
    tot_pair = sum(pair.values())
    out={}
    if tot_node == 0 or tot_pair == 0:
        return out
    for (a,b), c in pair.items():
        pa = node[a]/tot_node
        pb = node[b]/tot_node
        pab = c/tot_pair
        pmi = math.log((pab/(pa*pb)) + 1e-12)
        out[(a,b)] = pmi/(-math.log(pab + 1e-12))
    return out

def signed_diff_edges(pair_one, pair_rest, npmi_one, npmi_rest, min_total=5, top_k=70):
    feats = [f for f in set(pair_one)|set(pair_rest) if (pair_one.get(f,0)+pair_rest.get(f,0)) >= min_total]
    diffs=[]
    for f in feats:
        d = npmi_one.get(f,0.0) - npmi_rest.get(f,0.0)
        total = pair_one.get(f,0) + pair_rest.get(f,0)
        score = d * math.log(total+1)
        diffs.append((f,d,score,total))
    pos = sorted([x for x in diffs if x[2] > 0], key=lambda x:x[2], reverse=True)[:top_k]
    neg = sorted([x for x in diffs if x[2] < 0], key=lambda x:x[2])[:top_k]
    return pos, neg

def build_signed_graph(pos, neg, node_one, node_rest):
    G = nx.Graph()
    def add_edge(edge, d, score, total, side):
        a,b = edge
        f1a, fra = node_one.get(a,0), node_rest.get(a,0)
        f1b, frb = node_one.get(b,0), node_rest.get(b,0)
        G.add_node(a, freq=f1a+fra, f_one=f1a, f_rest=fra)
        G.add_node(b, freq=f1b+frb, f_one=f1b, f_rest=frb)
        G.add_edge(a,b, side=side, d=d, weight=abs(d), count=total, score=abs(score))
    for (edge,d,score,total) in pos:
        add_edge(edge,d,score,total,"ONE")
    for (edge,d,score,total) in neg:
        add_edge(edge,d,score,total,"REST")
    for n,data in G.nodes(data=True):
        f1, fr = data.get("f_one",0), data.get("f_rest",0)
        data["dominant"] = "ONE" if math.log((f1+1)/(fr+1)) > 0 else "REST"
    return G

def draw_signed_network(G, title, outpath, seed=7, one_color="#2b6cb0", rest_color="#a0aec0"):
    plt.figure(figsize=(11, 9))
    if G.number_of_nodes()==0 or G.number_of_edges()==0:
        plt.title(title, fontproperties=FP)
        plt.axis("off")
        plt.savefig(outpath, dpi=320, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(G, seed=seed, k=0.7, weight="weight")

    freqs = np.array([G.nodes[n].get("freq",1) for n in G.nodes()])
    node_sizes = 90 + 120*np.log1p(freqs)
    node_colors = [one_color if G.nodes[n].get("dominant")=="ONE" else rest_color for n in G.nodes()]

    edges = list(G.edges())
    weights = np.array([G.edges[e].get("weight",0.0) for e in edges])
    counts  = np.array([G.edges[e].get("count",1) for e in edges])

    widths = 0.6 + 4.0*(weights-weights.min())/(weights.max()-weights.min()+1e-9)
    widths = widths*(0.7+0.3*np.log1p(counts)/np.log1p(counts.max()))

    one_edges = [e for e in edges if G.edges[e].get("side")=="ONE"]
    rest_edges = [e for e in edges if G.edges[e].get("side")=="REST"]

    def widths_for(sub_edges):
        if not sub_edges:
            return []
        idx = [edges.index(e) for e in sub_edges]
        return [float(widths[i]) for i in idx]

    if rest_edges:
        nx.draw_networkx_edges(G, pos, edgelist=rest_edges, width=widths_for(rest_edges),
                               edge_color=rest_color, alpha=0.28)
    if one_edges:
        nx.draw_networkx_edges(G, pos, edgelist=one_edges, width=widths_for(one_edges),
                               edge_color=one_color, alpha=0.72)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.92)

    for n,(x,y) in pos.items():
        plt.text(x,y,n,fontproperties=FP,fontsize=11,ha="center",va="center")

    plt.title(title, fontproperties=FP)
    plt.axis("off")

    legend_elements = [
        Patch(facecolor=one_color, edgecolor=one_color, label="ONE(해당 관청)"),
        Patch(facecolor=rest_color, edgecolor=rest_color, label="REST(다른 3관청)"),
    ]
    plt.legend(handles=legend_elements, loc="lower left", frameon=False, prop=FP)
    plt.savefig(outpath, dpi=340, bbox_inches="tight")
    plt.close()

for t in targets:
    docs_one  = df.loc[df.office==t, "tokens"].tolist()
    docs_rest = df.loc[df.office.notna() & (df.office!=t), "tokens"].tolist()

    node_one, pair_one = cooc_counts(docs_one, window=5)
    node_rest, pair_rest = cooc_counts(docs_rest, window=5)

    npmi_one = edge_npmi(node_one, pair_one)
    npmi_rest = edge_npmi(node_rest, pair_rest)

    pos_edges, neg_edges = signed_diff_edges(pair_one, pair_rest, npmi_one, npmi_rest, min_total=5, top_k=70)
    G = build_signed_graph(pos_edges, neg_edges, node_one, node_rest)

    outpng = os.path.join(OUTDIR, f"signed_diff_office_{t}.png")
    draw_signed_network(G, f"{t} one-vs-rest 차등 네트워크(색상 구분)", outpng, seed=7)

print("DONE", OUTDIR)
