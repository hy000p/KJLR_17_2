# -*- coding: utf-8 -*-
"""
조선시대 계회시(엑셀 1번 시트: '계회시_저자') 1-gram(단일 한자) 분석 재현 코드

- 입력: 엑셀 파일(기본값: /mnt/data/조선시대 계회시 목록.xlsx) 첫 번째 sheet
- 핵심 분류:
  * 개인 vs 관청: '유형' 열에 '0'이 포함되면 개인으로 간주
  * 4개 관청(사간원, 사헌부, 승정원, 의금부): 개인 제외 후, '유형'이 해당 관청명으로 시작하면 해당 관청으로 간주
- 전처리:
  * 한자(CJK) + 아라비아 숫자(0-9)만 1-gram 토큰으로 사용
  * 기능어/수사(보수적) 제거 목록 적용
  * (관청 4종 비교에서는) 숫자 포함 분석 + 숫자 제외 분석을 모두 수행
- 산출물:
  * PNG 그래프 다수(모두 저장)
  * CSV 테이블(상위/전체 결과 저장)
"""

import os, re, math, zipfile
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) 경로 설정 (필요시 수정)
# -----------------------------
EXCEL_PATH = r"/mnt/data/조선시대 계회시 목록.xlsx"
SHEET_NAME = 0  # 첫 번째 sheet
OUT_DIR = r"/mnt/data/1gram_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) 폰트(한자 표시) 설정
# -----------------------------
plt.rcParams["font.family"] = ["Noto Serif CJK JP", "Noto Sans CJK JP", "NanumMyeongjo", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 2) 기능어/수사(보수적) 제거 목록
#    ※ 지침의 "숫자"는 기능어로 보지 않으므로 제거 목록에 포함하지 않음
# -----------------------------
STOPWORDS = set("之 其 而 以 於 于 也 矣 焉 兮 乎 哉 者 所 乃 與 及 且 為 曰 云 則 非 無 有 不 復 亦 又 更 可 何 我 吾 爾 汝 彼 此 是 斯 相 諸 各 皆 每".split())

# 숫자(지침에서 '제거하지 말 것' 목록) + (숫자 제외 분석에 사용할 확장 집합)
KEEP_NUMBERS = set("一 二 三 四 五 六 七 八 九 十 百 千 萬 兩 雙 第".split())
NUMERALS_EXCLUDE = set(list(KEEP_NUMBERS) + list("零 〇 廿 卅 卌 拾".split())) | set(list("0123456789"))

TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff0-9]")  # CJK + digits

def tokenize(text, remove_stopwords=True, exclude_numbers=False):
    """원문 -> 1-gram 토큰 리스트"""
    if pd.isna(text):
        return []
    toks = TOKEN_PATTERN.findall(str(text))
    out = []
    for t in toks:
        if remove_stopwords and (t in STOPWORDS):
            continue
        if exclude_numbers and (t in NUMERALS_EXCLUDE):
            continue
        out.append(t)
    return out

def group_counts(df_sub, token_col="tokens"):
    c = Counter()
    total = 0
    for toks in df_sub[token_col]:
        c.update(toks)
        total += len(toks)
    return c, total

def log_odds_z(counts_a, n_a, counts_b, n_b, prior_counts, prior_scale=1000.0, min_total=5):
    """
    Monroe et al.(2008) 스타일의 log-odds ratio z-score(정보적 Dirichlet prior)
    - prior_counts로부터 P(w)를 추정하고, prior_scale(의사 표본 크기)로 α_w = prior_scale * P(w)
    - z는 (집단 규모 차이 + 희귀 단어 불안정성)을 동시에 완화하는 효과가 있음
    """
    vocab = set(prior_counts.keys()) | set(counts_a.keys()) | set(counts_b.keys())
    N_prior_raw = sum(prior_counts.values())
    # 경험적 prior 확률
    p = {w: (prior_counts.get(w, 0) / N_prior_raw) for w in vocab}
    alpha0 = float(prior_scale)

    rows = []
    for w in vocab:
        ca = counts_a.get(w, 0)
        cb = counts_b.get(w, 0)
        if ca + cb < min_total:
            continue
        aw = alpha0 * p[w] if p[w] > 0 else alpha0 / max(1, len(vocab))

        # logit 계산
        logit_a = math.log((ca + aw) / (n_a - ca + alpha0 - aw))
        logit_b = math.log((cb + aw) / (n_b - cb + alpha0 - aw))
        delta = logit_a - logit_b

        var = 1.0 / (ca + aw) + 1.0 / (cb + aw)
        z = delta / math.sqrt(var)
        rows.append((w, ca, cb, delta, z))

    res = pd.DataFrame(rows, columns=["token", "count_a", "count_b", "log_odds", "z"])
    res.sort_values("z", ascending=False, inplace=True)
    return res

# -----------------------------
# 3) 그래프 유틸
# -----------------------------
def save_fig(path, dpi=200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_diverging_bar(tokens, values, title, xlabel, outpath):
    y = np.arange(len(tokens))
    plt.figure(figsize=(8, max(4, 0.28 * len(tokens) + 1.8)))
    ax = plt.gca()
    values = np.array(values)
    pos_mask = values >= 0
    neg_mask = ~pos_mask

    # 색을 직접 지정하지 않고(기본 컬러 사이클 사용) 양/음수 막대를 분리해서 그림
    ax.barh(y[pos_mask], values[pos_mask])
    ax.barh(y[neg_mask], values[neg_mask])
    ax.axvline(0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    save_fig(outpath)

def plot_dumbbell(df_sel, title, outpath):
    tokens = df_sel["token"].tolist()
    y = np.arange(len(tokens))
    plt.figure(figsize=(8, max(4, 0.28 * len(tokens) + 1.8)))
    ax = plt.gca()

    x_off = df_sel["rate_official_per1k"].values
    x_per = df_sel["rate_personal_per1k"].values

    for yi, a, b in zip(y, x_off, x_per):
        ax.plot([a, b], [yi, yi], linewidth=1)

    ax.plot(x_off, y, marker="o", linestyle="None", label="관청(1k당)")
    ax.plot(x_per, y, marker="o", linestyle="None", label="개인(1k당)")

    ax.set_yticks(y)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("빈도 (1,000 토큰당)")
    ax.legend()
    save_fig(outpath)

def plot_heatmap(mat, title, outpath, cbar_label="1,000 토큰당 빈도"):
    plt.figure(figsize=(7, max(4.5, 0.28 * len(mat) + 2)))
    ax = plt.gca()
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    save_fig(outpath)

def plot_top_bar(df_top, value_col, title, xlabel, outpath):
    tokens = df_top["token"].tolist()
    values = df_top[value_col].values
    y = np.arange(len(tokens))
    plt.figure(figsize=(7, max(4, 0.28 * len(tokens) + 1.6)))
    ax = plt.gca()
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    save_fig(outpath)

# -----------------------------
# 4) 데이터 로드 + 집단/관청 라벨링
# -----------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df["유형_str"] = df["유형"].astype(str).str.strip()
df["is_personal"] = df["유형_str"].str.contains("0")
df["group_personal_official"] = np.where(df["is_personal"], "개인", "관청")

df["office"] = None
for off in ["사간원", "사헌부", "승정원", "의금부"]:
    df.loc[(~df["is_personal"]) & (df["유형_str"].str.startswith(off)), "office"] = off

# 토큰화(기능어 제거 기본)
df["tokens"] = df["원문"].apply(lambda x: tokenize(x, remove_stopwords=True, exclude_numbers=False))
df["tokens_nonum"] = df["원문"].apply(lambda x: tokenize(x, remove_stopwords=True, exclude_numbers=True))

# -----------------------------
# 5) (A) 개인 vs 관청: 1,000토큰 정규화
# -----------------------------
def build_personal_official_tables(df_all, min_total=10):
    c_p, n_p = group_counts(df_all[df_all["group_personal_official"] == "개인"], "tokens")
    c_o, n_o = group_counts(df_all[df_all["group_personal_official"] == "관청"], "tokens")

    vocab = set(c_p) | set(c_o)
    rows = []
    for t in vocab:
        cp = c_p.get(t, 0)
        co = c_o.get(t, 0)
        tot = cp + co
        if tot < min_total:
            continue
        rp = cp / n_p * 1000
        ro = co / n_o * 1000
        rows.append((t, cp, co, tot, rp, ro, rp - ro))
    freq_df = pd.DataFrame(rows, columns=["token", "count_personal", "count_official", "count_total",
                                          "rate_personal_per1k", "rate_official_per1k", "diff_per1k"])
    return freq_df, (c_p, n_p, c_o, n_o)

freq_df, (counts_personal, n_personal, counts_official, n_official) = build_personal_official_tables(df, min_total=10)
freq_df.to_csv(os.path.join(OUT_DIR, "personal_vs_official_per1k_all_tokens.csv"), index=False, encoding="utf-8-sig")

# 상위 차이 토큰(양/음 각 15개) 추출
top_pos = freq_df.sort_values("diff_per1k", ascending=False).head(15)
top_neg = freq_df.sort_values("diff_per1k", ascending=True).head(15)
sel = pd.concat([top_pos, top_neg], axis=0).drop_duplicates("token").sort_values("diff_per1k", ascending=False)
sel.to_csv(os.path.join(OUT_DIR, "personal_vs_official_per1k_top.csv"), index=False, encoding="utf-8-sig")

plot_diverging_bar(
    sel["token"].tolist(),
    sel["diff_per1k"].tolist(),
    title="개인 vs 관청: 1-gram 빈도 차이(1,000토큰당, 기능어 제거)",
    xlabel="(개인 − 관청) 1,000토큰당 빈도 차이",
    outpath=os.path.join(OUT_DIR, "personal_vs_official_per1k_diverging.png")
)

# 절대빈도 비교(차이 큰 토큰 30개)
tmp = freq_df.copy()
tmp["absdiff"] = tmp["diff_per1k"].abs()
sel2 = tmp.sort_values("absdiff", ascending=False).head(30)
plot_dumbbell(
    sel2,
    title="개인 vs 관청: 상위 차이 1-gram의 절대 빈도(1,000토큰당, 기능어 제거)",
    outpath=os.path.join(OUT_DIR, "personal_vs_official_per1k_dumbbell.png")
)

# -----------------------------
# 6) (B) 개인 vs 관청: log-odds z (정규화)
# -----------------------------
prior_counts = Counter()
prior_counts.update(counts_personal)
prior_counts.update(counts_official)

lo_personal = log_odds_z(counts_personal, n_personal, counts_official, n_official,
                         prior_counts=prior_counts, prior_scale=1000, min_total=10)
lo_personal.to_csv(os.path.join(OUT_DIR, "personal_vs_official_logodds_z.csv"), index=False, encoding="utf-8-sig")

top_pos_z = lo_personal.head(15)
top_neg_z = lo_personal.sort_values("z", ascending=True).head(15)
lo_sel = pd.concat([top_pos_z, top_neg_z], axis=0).drop_duplicates("token").sort_values("z", ascending=False)

plot_diverging_bar(
    lo_sel["token"].tolist(),
    lo_sel["z"].tolist(),
    title="개인 vs 관청: 1-gram log-odds z (정보적 Dirichlet prior)",
    xlabel="z 점수 (양수: 개인 쪽, 음수: 관청 쪽)",
    outpath=os.path.join(OUT_DIR, "personal_vs_official_logodds_z_diverging.png")
)

# -----------------------------
# 7) (C) 4개 관청 비교: 숫자 포함/제외(각각) + per1k + log-odds z
# -----------------------------
df_off = df[df["office"].notna()].copy()
OFFICES = ["사간원", "사헌부", "승정원", "의금부"]

def office_distinctive_per1k(office_counts, office_totals, target_office, min_total=5):
    other_offs = [o for o in office_counts if o != target_office]
    counts_t = office_counts[target_office]
    n_t = office_totals[target_office]

    counts_o = Counter()
    n_o = 0
    for o in other_offs:
        counts_o.update(office_counts[o])
        n_o += office_totals[o]

    vocab = set(counts_t) | set(counts_o)
    rows = []
    for w in vocab:
        ct = counts_t.get(w, 0)
        co = counts_o.get(w, 0)
        if ct + co < min_total:
            continue
        rt = ct / n_t * 1000
        ro = co / n_o * 1000
        rows.append((w, ct, co, rt, ro, rt - ro))

    res = pd.DataFrame(rows, columns=["token", "count_target", "count_other",
                                      "rate_target_per1k", "rate_other_per1k", "diff_per1k"])
    res.sort_values("diff_per1k", ascending=False, inplace=True)
    return res

def run_office_block(token_col, totals_col_suffix, tag):
    # 1) counts/totals
    office_counts = {}
    office_totals = {}
    for off in OFFICES:
        sub = df_off[df_off["office"] == off]
        c, t = group_counts(sub, token_col)
        office_counts[off] = c
        office_totals[off] = t

    # 2) per-1000 heatmap(변동 큰 토큰)
    vocab = set()
    for c in office_counts.values():
        vocab |= set(c.keys())

    rows = []
    for tok in vocab:
        total = sum(office_counts[off].get(tok, 0) for off in OFFICES)
        if total < 8:
            continue
        rates = {off: office_counts[off].get(tok, 0) / office_totals[off] * 1000 for off in OFFICES}
        rows.append({"token": tok, "total": total, **{f"rate_{off}": rates[off] for off in OFFICES},
                     "var": np.var(list(rates.values()))})

    mat_df = pd.DataFrame(rows).sort_values("var", ascending=False)
    mat_df.to_csv(os.path.join(OUT_DIR, f"offices_per1k_matrix_{tag}.csv"), index=False, encoding="utf-8-sig")

    top_tokens = mat_df.head(35)["token"].tolist()
    hm = mat_df.set_index("token")[[f"rate_{off}" for off in OFFICES]].loc[top_tokens]
    hm.columns = OFFICES

    plot_heatmap(
        hm,
        title=f"4개 관청: 변동 큰 1-gram의 빈도(1,000토큰당, {tag})",
        outpath=os.path.join(OUT_DIR, f"offices_per1k_heatmap_{tag}.png"),
        cbar_label="1,000 토큰당 빈도"
    )

    # 3) per-1000 distinctive bar (각 관청 vs 나머지 3개)
    for off in OFFICES:
        res = office_distinctive_per1k(office_counts, office_totals, off, min_total=5)
        res.to_csv(os.path.join(OUT_DIR, f"office_{off}_per1k_diff_{tag}.csv"), index=False, encoding="utf-8-sig")
        top_df = res.head(15)
        plot_top_bar(
            top_df, "diff_per1k",
            title=f"{off}: 특징적 1-gram (1,000토큰당 빈도 차이, {tag})",
            xlabel=f"{off} − (다른 3관청) 1,000토큰당 빈도 차이",
            outpath=os.path.join(OUT_DIR, f"office_{off}_per1k_distinctive_{tag}.png")
        )

    # 4) log-odds z (각 관청 vs 나머지 3개)
    prior_counts = Counter()
    for off in OFFICES:
        prior_counts.update(office_counts[off])

    office_lo = {}
    for off in OFFICES:
        counts_t = office_counts[off]
        n_t = office_totals[off]
        other = Counter()
        n_o = 0
        for o in OFFICES:
            if o == off:
                continue
            other.update(office_counts[o])
            n_o += office_totals[o]
        office_lo[off] = log_odds_z(counts_t, n_t, other, n_o,
                                    prior_counts=prior_counts, prior_scale=1000, min_total=5)
        office_lo[off].to_csv(os.path.join(OUT_DIR, f"office_{off}_logodds_z_{tag}.csv"), index=False, encoding="utf-8-sig")

        top_df = office_lo[off].head(15)
        plot_top_bar(
            top_df, "z",
            title=f"{off}: 특징적 1-gram (log-odds z, {tag})",
            xlabel="z 점수",
            outpath=os.path.join(OUT_DIR, f"office_{off}_logodds_z_{tag}.png")
        )

    # 5) log-odds z heatmap (각 관청 상위 10 토큰 union)
    union = set()
    for off in OFFICES:
        union |= set(office_lo[off].head(10)["token"])
    union = list(union)

    z_rows = []
    for tok in union:
        row = {"token": tok}
        for off in OFFICES:
            # 빠른 lookup
            df_idx = office_lo[off].set_index("token")
            row[off] = df_idx.loc[tok, "z"] if tok in df_idx.index else np.nan
        z_rows.append(row)

    z_mat = pd.DataFrame(z_rows).set_index("token")[OFFICES]
    z_mat = z_mat.loc[z_mat.max(axis=1).sort_values(ascending=False).index]

    plot_heatmap(
        z_mat,
        title=f"4개 관청: log-odds z 특징어(상위 토큰 묶음, {tag})",
        outpath=os.path.join(OUT_DIR, f"offices_logodds_z_heatmap_{tag}.png"),
        cbar_label="z 점수 (클수록 해당 관청 쪽)"
    )

# 숫자 포함/제외 각각 실행
run_office_block(token_col="tokens", totals_col_suffix="", tag="with_numbers")
run_office_block(token_col="tokens_nonum", totals_col_suffix="_nonum", tag="no_numbers")

# -----------------------------
# 8) 결과물 zip 묶기(선택)
# -----------------------------
zip_path = os.path.join(os.path.dirname(OUT_DIR), "1gram_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            arc = os.path.relpath(fp, OUT_DIR)
            z.write(fp, arcname=arc)

print("DONE. Outputs:", OUT_DIR)
print("ZIP:", zip_path)
