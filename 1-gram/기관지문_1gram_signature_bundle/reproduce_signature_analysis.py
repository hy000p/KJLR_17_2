# -*- coding: utf-8 -*-
"""
조선시대 계회시(4개 관청) '기관 지문(signature)' 1-gram 분석
- 입력: /mnt/data/조선시대 계회시 목록.xlsx (sheet: 계회시_저자)
- 출력: PNG 3종 + CSV 3종 (재생산 가능)

핵심 아이디어
1) 1-gram(한자 1자) 빈도 기반
2) 기능어 제거(보수적 목록) + (옵션) 관청 별칭 보호(吾, 相)
3) log-odds ratio with informative Dirichlet prior (Monroe et al., 2008)로
   '각 기관에서 상대적으로 과대표현된 토큰'을 기관 지문으로 추출
"""

import os
import math
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

import regex as re2  # pip install regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


# =========================
# 0) 사용자 설정
# =========================
EXCEL_PATH = "/mnt/data/조선시대 계회시 목록.xlsx"
SHEET_NAME = "계회시_저자"

OUTPUT_DIR = "/mnt/data/기관지문_1gram_signature"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OFFICES = ["사헌부", "사간원", "승정원", "의금부"]

# 기능어/수사 제거 목록(사용자 제공, '보수적')
STOPWORDS = "之 其 而 以 於 于 也 矣 焉 兮 乎 哉 者 所 乃 與 及 且 為 曰 云 則 非 無 有 不 復 亦 又 更 可 何 我 吾 爾 汝 彼 此 是 斯 相 諸 各 皆 每".split()

# (주의) 숫자는 제거하지 않음(사용자 제공)
NUMERALS = "一 二 三 四 五 六 七 八 九 十 百 千 萬 兩 雙 第".split()

# 별칭 보존 옵션:
# - 의금부 별칭 '金吾'의 '吾', 사헌부 별칭 '相臺'의 '相'은 기능어 목록에 들어가 있으나
#   기관 지문 추출에서는 중요한 표지일 수 있으므로, 필요 시 보존합니다.
PRESERVE_ALIAS_CHARS = True
KEEP_FROM_STOP = set(list("吾相"))  # 별칭에서 중요도가 큰 기능어성 한자


# 시각화 폰트: CJK 지원 폰트 지정(환경에 따라 경로 수정 가능)
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# =========================
# 1) 유틸 함수
# =========================
def setup_matplotlib_font():
    """CJK(한자) 렌더링을 위해 폰트 설정."""
    if os.path.exists(CJK_FONT_PATH):
        font_manager.fontManager.addfont(CJK_FONT_PATH)
        mpl.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()
    mpl.rcParams["axes.unicode_minus"] = False


def tokenize_han(text: str, stop_set: set) -> list:
    """원문에서 한자(=Han script)만 1-gram으로 추출하고 기능어 제거."""
    chars = re2.findall(r"\p{Han}", str(text))
    return [c for c in chars if c not in stop_set]


def log_odds_z(counts_a: Counter, counts_b: Counter, prior_counts: Counter,
               min_total: int = 2, eps: float = 0.01) -> pd.DataFrame:
    """
    log-odds ratio with informative Dirichlet prior (Monroe et al., 2008)
    - counts_a: 집단 A 토큰 카운트
    - counts_b: 집단 B 토큰 카운트
    - prior_counts: 비교 말뭉치(여기서는 4기관 전체)의 토큰 카운트(정보적 사전분포)
    - min_total: (A+B) 합계 최소 등장 횟수 필터
    """
    vocab = set(prior_counts.keys()) | set(counts_a.keys()) | set(counts_b.keys())
    alpha0 = sum(prior_counts.get(w, 0) for w in vocab)
    n_a = sum(counts_a.get(w, 0) for w in vocab)
    n_b = sum(counts_b.get(w, 0) for w in vocab)

    rows = []
    for w in vocab:
        y_a = counts_a.get(w, 0)
        y_b = counts_b.get(w, 0)
        if (y_a + y_b) < min_total:
            continue

        alpha_w = prior_counts.get(w, 0) + eps  # 0 방지용 eps
        logit_a = math.log((y_a + alpha_w) / (n_a + alpha0 - (y_a + alpha_w)))
        logit_b = math.log((y_b + alpha_w) / (n_b + alpha0 - (y_b + alpha_w)))

        delta = logit_a - logit_b
        var = 1 / (y_a + alpha_w) + 1 / (y_b + alpha_w)
        z = delta / math.sqrt(var)

        rows.append((w, y_a, y_b, delta, z))

    df = pd.DataFrame(rows, columns=["token", "count_a", "count_b", "delta", "z"])
    df.sort_values("z", ascending=False, inplace=True)
    return df


def find_snippets(df: pd.DataFrame, token: str, n: int = 3, window: int = 12) -> list:
    """토큰이 등장하는 원문 일부(±window) 스니펫을 n개 추출."""
    examples = []
    for _, row in df.iterrows():
        text = str(row["원문"])
        idx = text.find(token)
        if idx != -1:
            start = max(0, idx - window)
            end = min(len(text), idx + window + 1)
            snippet = text[start:end].replace("\n", " ")
            examples.append({
                "office": row["office"],
                "제목": row["제목"],
                "문인": row["문인"],
                "문체": row["문체"],
                "snippet": snippet
            })
            if len(examples) >= n:
                break
    return examples


# =========================
# 2) 데이터 로딩 및 전처리
# =========================
def load_data() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 4기관 필터
    typ = df["유형"].astype(str).str.strip()
    df = df[typ.str.startswith(tuple(OFFICES))].copy()

    # office 라벨 추출
    df["office"] = df["유형"].astype(str).str.strip().str.extract(r"^(사헌부|사간원|승정원|의금부)")[0]
    return df


def build_stop_set() -> set:
    stop = set(STOPWORDS)
    # 숫자는 제거하지 않음(여기서는 STOPWORDS에 숫자가 없지만, 안전을 위해)
    stop = stop - set(NUMERALS)

    if PRESERVE_ALIAS_CHARS:
        stop = stop - KEEP_FROM_STOP
    return stop


# =========================
# 3) 분석 + 시각화
# =========================
def main():
    setup_matplotlib_font()

    df = load_data()
    stop_set = build_stop_set()

    # 토큰화
    df["tokens"] = df["원문"].map(lambda x: tokenize_han(x, stop_set))

    # 집단별 카운트
    group_counts = {}
    group_total = {}
    for off in OFFICES:
        toks = [t for toks in df[df.office == off]["tokens"] for t in toks]
        c = Counter(toks)
        group_counts[off] = c
        group_total[off] = sum(c.values())

    # prior: 4기관 전체
    prior = Counter()
    for off in OFFICES:
        prior.update(group_counts[off])

    # log-odds z: 각 기관 vs 나머지 3기관
    sig = {}
    for off in OFFICES:
        others = Counter()
        for o2 in OFFICES:
            if o2 != off:
                others.update(group_counts[o2])
        sig[off] = log_odds_z(group_counts[off], others, prior, min_total=2)

    # 3-1) 막대그래프 (2x2)
    top_k = 12
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    sig_tables = []
    top_tokens = {}

    for ax, off in zip(axes, OFFICES):
        df_sig = sig[off].copy()
        df_sig = df_sig[df_sig["count_a"] >= 3].head(top_k)
        top_tokens[off] = df_sig["token"].tolist()

        # 저장용 표
        tmp = df_sig.copy()
        tmp["office"] = off
        sig_tables.append(tmp)

        # 시각화용(역순)
        plot_df = df_sig.iloc[::-1]
        ax.barh(plot_df["token"], plot_df["z"])
        ax.set_title(f"{off} 1-gram 기관 지문(상위 {top_k}): log-odds z")
        ax.set_xlabel("log-odds z (vs 다른 3기관)")
        for i, (z, c) in enumerate(zip(plot_df["z"], plot_df["count_a"])):
            ax.text(z, i, f" {c}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "signature_bar_4offices.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3-2) 히트맵 (빈도/1000자)
    top_n = 8
    union = []
    for off in OFFICES:
        toks = sig[off]
        toks = toks[toks["count_a"] >= 3].head(top_n)["token"].tolist()
        union.extend(toks)

    seen = set()
    union_unique = []
    for t in union:
        if t not in seen:
            union_unique.append(t)
            seen.add(t)

    mat = []
    for tok in union_unique:
        row = []
        for off in OFFICES:
            row.append(group_counts[off].get(tok, 0) / group_total[off] * 1000)
        mat.append(row)
    mat = np.array(mat)

    fig = plt.figure(figsize=(9, max(6, 0.35 * len(union_unique))))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(range(len(union_unique)))
    ax.set_yticklabels(union_unique)
    ax.set_xticks(range(len(OFFICES)))
    ax.set_xticklabels(OFFICES)
    ax.set_title("기관 지문 토큰(1-gram) 빈도 히트맵 (토큰당 1000자 기준)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("빈도 / 1000자")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "signature_heatmap_per1000.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3-3) 문헌(작품) 단위 분포: TF-IDF PCA
    docs = [" ".join(toks) for toks in df["tokens"]]
    vectorizer = TfidfVectorizer(min_df=2, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(docs)

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X.toarray())

    df_pca = pd.DataFrame({
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
        "office": df["office"].values
    })
    df_pca.to_csv(os.path.join(OUTPUT_DIR, "doc_pca_coordinates.csv"), index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(9, 7))
    ax = plt.gca()
    for off in OFFICES:
        sub = df_pca[df_pca.office == off]
        ax.scatter(sub.PC1, sub.PC2, label=off, alpha=0.75)  # 색은 matplotlib 기본 cycle 사용
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% 분산)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% 분산)")
    ax.set_title("문헌(작품) 단위 1-gram TF-IDF PCA: 4기관 분포")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "doc_pca_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3-4) 결과표(CSV)
    sig_table = pd.concat(sig_tables, ignore_index=True)
    sig_table.to_csv(os.path.join(OUTPUT_DIR, "signature_top_tokens_logodds.csv"),
                     index=False, encoding="utf-8-sig")

    # 3-5) 스니펫(근거) 테이블
    rows = []
    for off in OFFICES:
        toks = top_tokens[off]
        dfo = df[df.office == off]
        for tok in toks:
            exs = find_snippets(dfo, tok, n=3, window=12)
            for ex in exs:
                rows.append({"office": off, "token": tok, **ex})
    snip_df = pd.DataFrame(rows)
    snip_df.to_csv(os.path.join(OUTPUT_DIR, "signature_token_snippets.csv"),
                   index=False, encoding="utf-8-sig")



    # 3-6) 기관 별칭/관서 표지 한자 '자기지시성' 밀도(보조 지표)
    #     - 각 기관 별칭들(예: 사헌부=驄馬/烏臺/憲府..., 의금부=金吾...)을 구성하는 1-gram 한자 집합을 만들고,
    #       해당 한자가 얼마나 자주 등장하는지(/1000자)로 계산합니다.
    alias_chars = {
        "사헌부": set(list("驄馬憲府栢府相臺烏臺御史臺監察司臺官大官柏")),
        "사간원": set(list("薇院諫院臺諫")),
        "승정원": set(list("銀臺政院喉院代言司")),
        "의금부": set(list("金吾禁府王府詔獄")),
    }
    # Han 문자만 남김
    alias_chars = {k: set([c for c in v if re2.match(r"\\p{Han}", c)]) for k, v in alias_chars.items()}

    cov_rows = []
    for off in OFFICES:
        total = group_total[off]
        alias_count = sum(group_counts[off].get(c, 0) for c in alias_chars[off])
        cov_rows.append({
            "office": off,
            "alias_count": alias_count,
            "total_tokens": total,
            "alias_per1000": alias_count / total * 1000,
            "alias_pct": alias_count / total * 100
        })

    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(os.path.join(OUTPUT_DIR, "alias_char_coverage.csv"), index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(cov_df["office"], cov_df["alias_per1000"])
    ax.set_ylabel("별칭/관서 표지 한자 빈도 (/1000자)")
    ax.set_title("기관 지문 보조지표: 관서 별칭·표지 한자 사용 밀도")
    for x, y in zip(cov_df["office"], cov_df["alias_per1000"]):
        ax.text(x, y, f"{y:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "alias_char_coverage_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Done. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
