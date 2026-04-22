# -*- coding: utf-8 -*-
"""
조선시대 계회시(첫 번째 sheet) 기반 2-gram 동시출현 의미네트워크
- 2-gram은 "문체(5/7/4/6언) 기반 + 행/구 경계 내부"에서만 추출
- 네트워크 엣지 가중치: NPMI × log-odds z (문서 단위 동시출현)
- 커뮤니티: Louvain(가중치) + 범례(대표 2-gram) 포함
- 결과: PNG 3종 저장
"""

import os, re, math
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import regex as reg
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx.algorithms.community as nx_comm


# =========================
# 0) 입력/출력 경로
# =========================
EXCEL_PATH = "조선시대 계회시 목록.xlsx"  # 실행 디렉터리에 파일이 있을 때
SHEET_NAME = 0  # 첫 번째 sheet
OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)


# =========================
# 1) 전처리 설정
# =========================
# 기능어/수사(보수적) 제거 목록 (단, 숫자류는 제거하지 않음)
STOPWORDS_STR = "之 其 而 以 於 于 也 矣 焉 兮 乎 哉 者 所 乃 與 及 且 為 曰 云 則 非 無 有 不 復 亦 又 更 可 何 我 吾 爾 汝 彼 此 是 斯 相 諸 各 皆 每"
NUMERALS_STR = "一 二 三 四 五 六 七 八 九 十 百 千 萬 兩 雙 第"
STOPWORDS = set(STOPWORDS_STR.split()) - set(NUMERALS_STR.split())

BOUNDARY_PATTERN = re.compile(r"[，,。\.、;；:：\?\؟!！\n\r\t]+")
HAN_PATTERN = reg.compile(r"\p{Han}")

def parse_style_len(style):
    """문체에서 4/5/6/7언 길이 추출(가능한 경우만)."""
    if pd.isna(style):
        return None
    m = re.search(r"(\d+)", str(style))
    if m:
        n = int(m.group(1))
        if n in (4, 5, 6, 7):
            return n
    return None

def split_into_chunks(chars, n):
    """n언 단위로 강제 분절(길이가 2 미만인 조각은 제외)."""
    return [chars[i:i+n] for i in range(0, len(chars), n) if len(chars[i:i+n]) >= 2]

def extract_bigrams_from_text(text, style_len=None):
    """
    2-gram 추출:
    1) 문장부호/개행 기준으로 1차 분절
    2) 분절 내부에서만 2-gram 추출 (경계 넘지 않음)
    3) 5/7/4/6언인 경우, 분절이 지나치게 길면 n언 단위로 추가 분절
    4) bigram 두 글자가 모두 기능어면 제외
    """
    if pd.isna(text):
        return []

    s = str(text).replace(" ", "").replace("\u3000", "")
    units = [u for u in BOUNDARY_PATTERN.split(s) if u]

    segments = []
    for u in units:
        chars = "".join(HAN_PATTERN.findall(u))
        if len(chars) < 2:
            continue

        if style_len in (4, 5, 6, 7):
            # 행 구분 부호가 누락되어 한 덩어리로 붙은 경우를 대비해 추가 분절
            if len(chars) > style_len and (len(chars) % style_len == 0 or len(chars) > style_len * 2):
                segments.extend(split_into_chunks(chars, style_len))
            else:
                segments.append(chars)
        else:
            segments.append(chars)

    bigrams = []
    for seg in segments:
        for i in range(len(seg) - 1):
            bg = seg[i:i+2]
            if (bg[0] in STOPWORDS) and (bg[1] in STOPWORDS):
                continue
            bigrams.append(bg)

    return bigrams


# =========================
# 2) NPMI + log-odds z
# =========================
def calc_npmi(pij, pi, pj):
    """NPMI (normalized PMI)."""
    if pij <= 0 or pi <= 0 or pj <= 0:
        return None
    pmi = math.log(pij / (pi * pj))
    return pmi / (-math.log(pij))

def logodds_z(a, b, c, d, alpha=0.5):
    """
    2x2 표 기반 log-odds ratio와 z.
    - a: i&j 동시 출현 문서 수
    - b: i만 출현
    - c: j만 출현
    - d: 둘 다 없음
    """
    a2, b2, c2, d2 = a + alpha, b + alpha, c + alpha, d + alpha
    lor = math.log((a2 * d2) / (b2 * c2))
    se = math.sqrt(1/a2 + 1/b2 + 1/c2 + 1/d2)
    z = lor / se
    return lor, z

def pastel_colors(k):
    """옅은 파스텔톤 팔레트(필요 수만큼)."""
    cols = []
    for cmap_name in ["Pastel1", "Pastel2", "Set3"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            cols.extend(list(cmap.colors))
        else:
            cols.extend([cmap(i) for i in np.linspace(0, 1, 12)])
    return cols[:k]


# =========================
# 3) 실행
# =========================
def main():
    # 폰트(한자/한글 깨짐 방지)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df = df[df["원문"].notna()].copy()

    # 작품별 2-gram 추출 + 문서단위 집합 생성
    doc_sets = []
    for _, row in df.iterrows():
        n = parse_style_len(row.get("문체"))
        bgs = extract_bigrams_from_text(row["원문"], n)
        if bgs:
            doc_sets.append(set(bgs))

    D = len(doc_sets)
    print(f"문서 수(D): {D}")

    # 문서빈도(dfreq)
    dfreq = Counter()
    for s in doc_sets:
        dfreq.update(s)

    # 네트워크 구성 파라미터(가독성 중심)
    MIN_DF = 6       # 문서빈도 하한(희귀 2-gram 과잉연결 방지)
    MIN_A = 3        # 동시출현 문서 수 하한
    MIN_NPMI = 0.15  # 양의 결속만
    MIN_Z = 3.0      # 통계적 강도 하한(대략 p<0.003 수준)

    vocab = [bg for bg, c in dfreq.items() if c >= MIN_DF]
    vocab_set = set(vocab)
    print(f"분석 대상 2-gram 수(문서빈도≥{MIN_DF}): {len(vocab)}")

    # 동시출현 카운트(문서 단위)
    co = Counter()
    for s in doc_sets:
        items = sorted(list(s & vocab_set))
        if len(items) < 2:
            continue
        for i, j in combinations(items, 2):
            co[(i, j)] += 1

    # 엣지 계산
    edges = []
    for (i, j), a in co.items():
        pi = dfreq[i] / D
        pj = dfreq[j] / D
        pij = a / D

        n = calc_npmi(pij, pi, pj)
        if n is None:
            continue

        b = dfreq[i] - a
        c = dfreq[j] - a
        d = D - (a + b + c)
        lor, z = logodds_z(a, b, c, d, alpha=0.5)

        score = z * n  # 결합 강도(가중치)
        edges.append((i, j, a, n, z, score, lor))

    # 필터링
    edges_f = [e for e in edges if (e[2] >= MIN_A and e[3] >= MIN_NPMI and e[4] >= MIN_Z)]
    edges_f.sort(key=lambda x: x[5], reverse=True)
    print(f"필터 후 엣지 수: {len(edges_f)}")

    # 그래프 생성
    G = nx.Graph()
    for i, j, a, n, z, score, lor in edges_f:
        G.add_edge(i, j, cooc=a, npmi=n, z=z, weight=score)

    # 연결성 확인: 가장 큰 연결요소만 시각화(논문용 가독성)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    Gc = G.subgraph(comps[0]).copy()
    print(f"시각화 대상 노드/엣지(최대 연결요소): {Gc.number_of_nodes()} / {Gc.number_of_edges()}")

    # Louvain 커뮤니티(가중치)
    comms = nx_comm.louvain_communities(Gc, weight="weight", seed=42)
    modularity = nx_comm.modularity(Gc, comms, weight="weight")
    print(f"Louvain 커뮤니티 수: {len(comms)}, modularity={modularity:.3f}")

    # 커뮤니티 id
    comm_id = {}
    for idx, c in enumerate(comms, start=1):
        for node in c:
            comm_id[node] = idx

    K = len(comms)
    cols = pastel_colors(K)

    # 가중차수(노드 중요도)
    wdeg = dict(Gc.degree(weight="weight"))

    # 커뮤니티 대표어(가중차수 상위 3개)
    rep = {}
    for cid, c in enumerate(comms, start=1):
        top = sorted(list(c), key=lambda n: wdeg.get(n, 0), reverse=True)[:3]
        rep[cid] = top

    # =========================
    # 3-1) 네트워크 그림
    # =========================
    pos = nx.spring_layout(Gc, weight="weight", seed=42, k=0.6 / np.sqrt(Gc.number_of_nodes()))

    sizes = np.array([wdeg[n] for n in Gc.nodes()])
    node_sizes = 200 + 2200 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
    node_colors = [cols[comm_id[n] - 1] for n in Gc.nodes()]

    edge_weights = np.array([Gc[u][v]["weight"] for u, v in Gc.edges()])
    edge_widths = 0.5 + 4 * (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)

    # 라벨: 전체 상위 25 + 커뮤니티 대표어
    top_global = [n for n, _ in sorted(wdeg.items(), key=lambda x: x[1], reverse=True)[:25]]
    label_nodes = set(top_global)
    for nodes in rep.values():
        label_nodes.update(nodes)

    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = plt.gca()
    ax.set_axis_off()

    nx.draw_networkx_edges(Gc, pos, ax=ax, width=edge_widths, alpha=0.25)
    nx.draw_networkx_nodes(
        Gc, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        linewidths=0.4, edgecolors="white", alpha=0.95
    )

    for n, (x, y) in pos.items():
        if n in label_nodes:
            fs = 7 + 7 * (wdeg[n] - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
            ax.text(x, y, n, fontsize=fs, ha="center", va="center", color="black")

    handles = []
    for cid in range(1, K + 1):
        lab = f"C{cid}: " + "·".join(rep[cid])
        handles.append(Patch(facecolor=cols[cid - 1], edgecolor="none", label=lab))

    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=9, title="커뮤니티(대표 2-gram)"
    )

    plt.title("2-gram 동시출현 의미네트워크 (NPMI × log-odds z 가중, Louvain 커뮤니티)", fontsize=16, pad=12)
    plt.tight_layout()

    out1 = os.path.join(OUTDIR, "semantic_network_bigram_logodds_npmi.png")
    plt.savefig(out1, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 3-2) 커뮤니티별 상위 2-gram 막대(2x5)
    # =========================
    comm_top = {}
    for cid, c in enumerate(comms, start=1):
        top = sorted([(n, wdeg[n]) for n in c], key=lambda x: x[1], reverse=True)[:10]
        comm_top[cid] = top

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), dpi=300)
    axes = axes.flatten()

    for cid, ax in zip(range(1, K + 1), axes):
        terms = [t for t, _ in comm_top[cid]][::-1]
        vals = [v for _, v in comm_top[cid]][::-1]
        ax.barh(range(len(terms)), vals, color=cols[cid - 1], alpha=0.9)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_title(f"C{cid}", fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[K:]:
        ax.axis("off")

    plt.suptitle("커뮤니티별 핵심 2-gram (가중차수 상위 10)", fontsize=16, y=1.02)
    plt.tight_layout()

    out2 = os.path.join(OUTDIR, "community_top_terms_bigram_network.png")
    plt.savefig(out2, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 3-3) 상위 엣지 20개
    # =========================
    top_edges = edges_f[:20]
    labels = [f"{i}-{j}" for i, j, *_ in top_edges][::-1]
    scores = [e[5] for e in top_edges][::-1]

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.gca()
    ax.barh(range(len(labels)), scores, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("결합 강도 (NPMI × log-odds z)", fontsize=11)
    ax.set_title("상위 결합(엣지) 20개", fontsize=14)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    out3 = os.path.join(OUTDIR, "top_edges_bigram_network.png")
    plt.savefig(out3, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
