# -*- coding: utf-8 -*-
"""
조선시대 계회시(엑셀 1st sheet) 원문 기반 2-gram(한자 2글자) 분석
- 반드시 '문체(4/5/6/7언)'를 반영해 행/구 경계 내부에서만 2-gram을 추출
- 기능어/수사(보수적) 제거(단, 숫자 한자는 제거하지 않음)
- (1) per-1K 빈도
- (2) log-odds z (독립성(문자 독립) 기반 기대치 대비 과대표현)
- (3) log-odds z × NPMI (과대표현 × 결속력) 결합지표
- 결과 그래프: PNG로 저장
"""

import os, re, math, collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# -----------------------
# 0) 입력/출력 설정
# -----------------------
EXCEL_PATH = "조선시대 계회시 목록.xlsx"  # 현재 스크립트와 같은 폴더에 두거나, 절대경로로 수정
SHEET_NAME = 0  # 첫 번째 sheet
OUT_DIR = "."   # 결과 저장 폴더
TOP_N = 30
MIN_COUNT_FOR_Z = 3

# -----------------------
# 1) 전처리 규칙
# -----------------------
STOP_CHARS = set("之 其 而 以 於 于 也 矣 焉 兮 乎 哉 者 所 乃 與 及 且 為 曰 云 則 非 無 有 不 復 亦 又 更 可 何 我 吾 爾 汝 彼 此 是 斯 相 諸 各 皆 每".split())
NUM_CHARS = set("一 二 三 四 五 六 七 八 九 十 百 千 萬 兩 雙 第".split())
# 숫자 한자는 기능어 목록에서 제외(=분석에 포함)
STOP_CHARS = STOP_CHARS - NUM_CHARS

PUNCT_PATTERN = re.compile(r"[。．\.！!？\?；;：:，,、\n\r\t\s]+")
# 한자(확장 포함) + 아라비아 숫자 + 〇/○
ALLOWED_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff0-9〇○]+")

def set_cjk_font():
    """한자 깨짐 방지를 위해 CJK 폰트 설정(환경에 따라 자동 탐색)."""
    preferred = ["Noto Sans CJK JP", "Noto Serif CJK JP", "NanumGothic", "AppleGothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    mpl.rcParams["axes.unicode_minus"] = False
    return None

def normalize_style(val):
    """문체(4/5/6/7언 등)를 정수(4,5,6,7)로 변환."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    mapping = {"오언":5, "칠언":7, "사언":4, "육언":6}
    for k,v in mapping.items():
        if k in s:
            return v
    m = re.search(r"([0-9])\s*언", s)
    if m:
        return int(m.group(1))
    m = re.search(r"([4567])", s)
    if m and "언" in s:
        return int(m.group(1))
    return None

def extract_lines(text, line_len=None):
    """
    원문에서 '행/구 경계'를 만들기:
    1) 우선 구두점 기반 분절(，。 등)
    2) 문체가 4/5/6/7언이고, 분절 결과가 해당 길이의 배수라면 -> 고정 길이로 chunking
       (구두점이 누락된 경우에도 '문체 기반 경계'를 최대한 반영)
    """
    if pd.isna(text):
        return []
    s = str(text).replace("\u3000", " ")
    parts = [p for p in PUNCT_PATTERN.split(s) if p]
    lines = []
    for part in parts:
        chars = "".join(ALLOWED_PATTERN.findall(part))
        if not chars:
            continue
        if line_len in (4,5,6,7) and len(chars) >= 2*line_len and len(chars) % line_len == 0:
            for i in range(0, len(chars), line_len):
                lines.append(chars[i:i+line_len])
        else:
            lines.append(chars)
    return lines

def extract_counts(df):
    """행/구 내부 2-gram + (기대치 계산용) 1-gram 집계."""
    bigram_counts = collections.Counter()
    unigram_counts = collections.Counter()
    for _, row in df.iterrows():
        line_len = normalize_style(row.get("문체"))
        for line in extract_lines(row.get("원문"), line_len=line_len):
            chars = list(line)
            for ch in chars:
                unigram_counts[ch] += 1
            for i in range(len(chars)-1):
                bg = chars[i] + chars[i+1]
                bigram_counts[bg] += 1
    return bigram_counts, unigram_counts

def filter_counts(bigram_counts, unigram_counts, min_count=1):
    """기능어 제거(단 숫자 한자는 유지)."""
    bg_f = collections.Counter()
    for bg,c in bigram_counts.items():
        if c < min_count:
            continue
        if bg[0] in STOP_CHARS or bg[1] in STOP_CHARS:
            continue
        bg_f[bg] = c
    uni_f = collections.Counter({ch:c for ch,c in unigram_counts.items() if ch not in STOP_CHARS})
    return bg_f, uni_f

def compute_metrics(bg_counts, uni_counts, alpha_bg=0.01, alpha_uni=0.01, min_count=3):
    """
    (2) log-odds z (단일 코퍼스 버전):
    - '문자 독립(Independence)' 가정에서의 기대 빈도(=p(a)*p(b))를 내부 기준선으로 삼고,
      관측 빈도가 그 기대치보다 얼마나 과대표현되는지 log-odds 차이를 z로 표준화.
    - 집단 비교가 아니라 '내부 기준선 대비 과대표현' 랭킹 지표로 해석.

    (3) 결합지표:
    - NPMI로 '결속력(콜로케이션 강도)'를 측정하고,
    - z × max(NPMI,0)로 "과대표현 + 결속력"을 동시에 큰 항목을 강조.
    """
    N = sum(bg_counts.values())
    U = sum(uni_counts.values())
    V = len(bg_counts)
    Vu = len(uni_counts)

    denom_u = U + alpha_uni * Vu
    p_uni = {ch:(c + alpha_uni)/denom_u for ch,c in uni_counts.items()}

    rows = []
    denom_bg = N + alpha_bg * V

    for bg, y in bg_counts.items():
        if y < min_count:
            continue
        a, b = bg[0], bg[1]
        pa = p_uni.get(a, alpha_uni/denom_u)
        pb = p_uni.get(b, alpha_uni/denom_u)
        p_exp = pa * pb
        x = N * p_exp  # 기대 count (독립성 가정)

        p1 = (y + alpha_bg) / denom_bg
        p2 = (x + alpha_bg) / denom_bg
        p1 = min(max(p1, 1e-12), 1-1e-12)
        p2 = min(max(p2, 1e-12), 1-1e-12)

        lo1 = math.log(p1/(1-p1))
        lo2 = math.log(p2/(1-p2))
        delta = lo1 - lo2
        var = 1/(y + alpha_bg) + 1/(x + alpha_bg)
        z = delta / math.sqrt(var)

        # NPMI(해석용): smoothing 없이 계산
        p_ab = y / N
        pa0 = uni_counts.get(a, 0) / U
        pb0 = uni_counts.get(b, 0) / U
        if p_ab > 0 and pa0 > 0 and pb0 > 0:
            pmi = math.log(p_ab / (pa0 * pb0))
            npmi = pmi / (-math.log(p_ab))
        else:
            npmi = float("nan")

        comb = z * max(npmi, 0) if not math.isnan(npmi) else float("nan")

        rows.append((bg, y, 1000*y/N, z, npmi, comb, x))

    return pd.DataFrame(rows, columns=["bigram","count","per1k","logodds_z","npmi","z_x_npmi","expected_count"])

def plot_bar(data, x_col, y_col, title, xlabel, filename, top_n=30):
    d = data.copy().head(top_n)
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28*len(d) + 1)))
    ax.barh(d[y_col], d[x_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    for i, val in enumerate(d[x_col].values):
        ax.text(val, i, f" {val:.2f}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_scatter(metrics_df, filename, n_annotate=12):
    d = metrics_df.dropna(subset=["logodds_z","npmi"]).copy()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(d["npmi"], d["logodds_z"], s=15, alpha=0.6)
    ax.set_xlabel("NPMI (결속력)")
    ax.set_ylabel("log-odds z (과대표현)")
    ax.set_title("2-gram의 과대표현(z) vs 결속력(NPMI)")
    top = d.sort_values("z_x_npmi", ascending=False).head(n_annotate)
    for _, r in top.iterrows():
        ax.text(r["npmi"], r["logodds_z"], r["bigram"], fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_cjk_font()

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    bigram_counts, unigram_counts = extract_counts(df)
    bg_f, uni_f = filter_counts(bigram_counts, unigram_counts, min_count=1)

    N = sum(bg_f.values())

    # (1) per-1K
    df_per1k = pd.DataFrame([(bg, c, 1000*c/N) for bg,c in bg_f.most_common(60)],
                            columns=["bigram","count","per1k"])
    df_per1k["label"] = df_per1k["bigram"]

    # (2)(3) log-odds z / z×NPMI
    metrics = compute_metrics(bg_f, uni_f, min_count=MIN_COUNT_FOR_Z)
    df_z = metrics.sort_values("logodds_z", ascending=False).head(60).copy()
    df_z["label"] = df_z["bigram"]
    df_comb = metrics.sort_values("z_x_npmi", ascending=False).head(60).copy()
    df_comb["label"] = df_comb["bigram"]

    # 그래프 저장
    plot_bar(df_per1k, "per1k", "label",
             "2-gram 빈도 상위(정규화: per 1,000 bigrams)",
             "per 1,000 bigrams",
             "bigram_per1k_top30.png", top_n=TOP_N)

    plot_bar(df_z, "logodds_z", "label",
             "2-gram 과대표현(로그오즈 z, 독립성 기준) 상위",
             "log-odds z (vs independence baseline)",
             "bigram_logodds_z_top30.png", top_n=TOP_N)

    plot_bar(df_comb, "z_x_npmi", "label",
             "2-gram 결합지표(로그오즈 z × NPMI) 상위",
             "z × NPMI",
             "bigram_logodds_z_npmi_top30.png", top_n=TOP_N)

    plot_scatter(metrics, "bigram_z_vs_npmi_scatter.png", n_annotate=12)

    # 표 저장(논문용 후처리 편의)
    df_per1k.to_csv(os.path.join(OUT_DIR, "bigram_per1k_top60.csv"), index=False, encoding="utf-8-sig")
    metrics.sort_values("logodds_z", ascending=False).head(200).to_csv(
        os.path.join(OUT_DIR, "bigram_logodds_z_top200.csv"), index=False, encoding="utf-8-sig")
    metrics.sort_values("z_x_npmi", ascending=False).head(200).to_csv(
        os.path.join(OUT_DIR, "bigram_z_npmi_top200.csv"), index=False, encoding="utf-8-sig")

    # 콘솔 출력(상위 10)
    print("\n[per-1K TOP10]")
    print(df_per1k.head(10)[["bigram","count","per1k"]].to_string(index=False))
    print("\n[log-odds z TOP10]")
    print(df_z.head(10)[["bigram","count","per1k","logodds_z","npmi","expected_count"]].to_string(index=False))
    print("\n[z×NPMI TOP10]")
    print(df_comb.head(10)[["bigram","count","per1k","logodds_z","npmi","z_x_npmi","expected_count"]].to_string(index=False))

if __name__ == "__main__":
    main()
