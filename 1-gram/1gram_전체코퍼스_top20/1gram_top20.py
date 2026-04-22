# -*- coding: utf-8 -*-
"""
조선시대 계회시(엑셀 첫 번째 sheet: '계회시_저자') 원문('원문' 열) 기반
- 기능어(수사) 제거
- 1-gram(한자 1글자) 빈도 상위 20 시각화
- PNG 저장 + 상위 20 표 CSV 저장

실행:
    python 1gram_top20.py
"""

import os
import re
import unicodedata
from collections import Counter

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# 1) 사용자 설정
# =========================
INPUT_EXCEL_PATH = r"/mnt/data/조선시대 계회시 목록.xlsx"   # 필요시 수정
SHEET_NAME = 0  # "첫 번째 sheet" = 0
TEXT_COL = "원문"

OUTPUT_DIR = r"/mnt/data/1gram_top20_outputs"  # 필요시 수정
PNG_NAME = "1gram_top20_excluding_function_words.png"
CSV_NAME = "1gram_top20_table.csv"

TOP_N = 20


# =========================
# 2) 기능어(수사) 제거 목록
#    ※ 숫자(一二三… 등)는 제거하지 말 것!
# =========================
STOPWORDS = set("之 其 而 以 於 于 也 矣 焉 兮 乎 哉 者 所 乃 與 及 且 為 曰 云 則 非 無 有 不 復 亦 又 更 可 何 我 吾 爾 汝 彼 此 是 斯 相 諸 各 皆 每".split())


# =========================
# 3) 토큰화(한자 1글자)
#    - CJK 통합한자 범위만 추출(구두점/공백/기타 제거)
# =========================
CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')

def iter_unigrams(text: str):
    text = unicodedata.normalize("NFKC", str(text))
    for ch in text:
        if CJK_RE.match(ch):
            yield ch


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 데이터 로드
    df = pd.read_excel(INPUT_EXCEL_PATH, sheet_name=SHEET_NAME)

    if TEXT_COL not in df.columns:
        raise KeyError(f"'{TEXT_COL}' 열을 찾을 수 없습니다. 실제 열 이름: {list(df.columns)}")

    texts = df[TEXT_COL].dropna().astype(str).tolist()

    # 빈도 계산
    counter = Counter()
    for t in texts:
        for tok in iter_unigrams(t):
            if tok in STOPWORDS:
                continue
            counter[tok] += 1

    top = counter.most_common(TOP_N)
    top_df = pd.DataFrame(top, columns=["1-gram", "빈도"])

    # 표 저장
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    top_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # =========================
    # 4) 그래프(가독성을 위해 가로막대)
    #    - 한글 UI(제목/축)용 폰트: NanumGothic
    #    - 한자 라벨용 폰트: Noto Sans CJK JP (환경에 설치되어 있는 경우)
    # =========================
    mpl.rcParams["font.family"] = "NanumGothic"
    mpl.rcParams["axes.unicode_minus"] = False

    tokens = [t for t, _ in top]
    counts = [c for _, c in top]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(tokens))[::-1], counts)
    plt.yticks(range(len(tokens))[::-1], tokens, fontname="Noto Sans CJK JP", fontsize=12)

    plt.xlabel("빈도(출현 횟수)")
    plt.title(f"기능어 제외 1-gram 상위 {TOP_N} (전체)")

    # 값 주석
    for i, cnt in enumerate(counts):
        y = len(tokens) - 1 - i
        plt.text(cnt, y, f" {cnt}", va="center", fontsize=9)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, PNG_NAME)
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("완료!")
    print(f"- PNG: {png_path}")
    print(f"- CSV: {csv_path}")


if __name__ == "__main__":
    main()
