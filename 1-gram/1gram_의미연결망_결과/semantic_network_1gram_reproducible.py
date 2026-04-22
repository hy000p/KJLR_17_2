# -*- coding: utf-8 -*-
"""
조선시대 계회시(엑셀 첫 번째 sheet) 기반 1-gram(한자 1자) 의미연결망(동시출현 네트워크) 재현 코드

- 개인(유형에 '0' 포함) vs 관청
- 4개 관청(사헌부/사간원/승정원/의금부)별 네트워크
- 차등(difference-only) 네트워크: NPMI(결합 강도) 차이 기반

산출물(OUTDIR):
- network_private.png, network_official.png
- diff_private_over_official.png, diff_official_over_private.png
- network_office_사헌부.png ... (4개)
- diff_office_사헌부_over_rest.png ... (4개)
- office_diff_top_edges.csv (관청별 차등 엣지 상위 15)
"""

import os, re, math, zipfile
from collections import Counter

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# =========================
# 0) 경로 설정
# =========================
EXCEL_PATH = r"/mnt/data/조선시대 계회시 목록.xlsx"   # 필요시 수정
OUTDIR     = r"/mnt/data/1gram_의미연결망_재현결과"   # 필요시 수정
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# 1) 폰트 설정 (한자/한글 깨짐 방지)
# =========================
# 본 환경에서는 Noto CJK 폰트가 설치되어 있어 이를 사용합니다.
# 다른 환경에서는 아래 FONT_PATH를 해당 폰트 경로로 바꾸세요.
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FP = fm.FontProperties(fname=FONT_PATH)
mpl.rcParams["axes.unicode_minus"] = False

# =========================
# 2) 데이터 로드 (첫 번째 sheet)
# =========================
df = pd.read_excel(EXCEL_PATH, sheet_name=0)
if not {"유형","원문"}.issubset(set(df.columns)):
    raise ValueError(f"필수 열(유형, 원문)이 없습니다. 현재 열: {df.columns.tolist()}")

# =========================
# 3) 전처리: 1-gram 토큰화 + 기능어 제거(보수적)
#    (단, 관청 별칭에 포함된 기능어 글자는 예외적으로 유지: 吾, 相 등)
# =========================
stopwords = set(list("之其而以於于也矣焉兮乎哉者所乃與及且為曰云則非無有不復亦又更可何我吾爾汝彼此是斯相諸各皆每"))
numerals  = set(list("一二三四五六七八九十百千萬兩雙第"))  # 제거하지 말 것
stopwords = stopwords - numerals

aliases = {
  "사헌부": ["驄馬","憲府","栢府","相臺","烏臺","御史臺","監察司","大관","臺官"],
  "사간원": ["薇院","諫院","臺諫","대간"],
  "승정원": ["銀臺","政院","喉院","代言司","대언사"],
  "의금부": ["金吾","禁府","王府","詔獄","金部"],
}
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

# =========================
# 4) 집단 라벨
# =========================
df["is_private"] = df["유형"].astype(str).str.contains("0", na=False)
df["group_pc"] = np.where(df["is_private"], "개인", "관청")

targets = ["사헌부","사간원","승정원","의금부"]
types = df["유형"].astype(str).fillna("").str.strip()
df["office"] = None
for t in targets:
    df.loc[types.str.startswith(t), "office"] = t

# =========================
# 5) 동시출현 네트워크 구성
# =========================
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
    for (a,b), c in pair.items():
        pa = node[a]/tot_node
        pb = node[b]/tot_node
        pab = c/tot_pair
        pmi = math.log((pab/(pa*pb)) + 1e-12)
        out[(a,b)] = pmi/(-math.log(pab + 1e-12))
    return out

def build_graph(node, pair, top_nodes=70, min_node=8, min_pair=3, top_edges=220):
    nodes=[w for w,c in node.most_common() if c>=min_node][:top_nodes]
    nodes_set=set(nodes)
    npmi=edge_npmi(node,pair)
    edges=[]
    for (a,b), c in pair.items():
        if c<min_pair or a not in nodes_set or b not in nodes_set:
            continue
        w=npmi.get((a,b),0.0)
        score=w*math.log(c+1)
        edges.append((a,b,c,w,score))
    edges=sorted(edges, key=lambda x:x[4], reverse=True)[:top_edges]
    G=nx.Graph()
    for n in nodes:
        G.add_node(n, freq=node[n])
    for a,b,c,w,score in edges:
        G.add_edge(a,b, weight=w, count=c, score=score)
    return G, npmi

def build_graph_auto(node,pair, top_nodes=70):
    if len(node)==0:
        return nx.Graph(), {}
    counts=list(node.values())
    min_node=int(np.percentile(counts,80))
    min_node=max(3, min(min_node,8))
    return build_graph(node,pair, top_nodes=top_nodes, min_node=min_node, min_pair=2, top_edges=180)

def draw_network(G, title, outpath, seed=7):
    plt.figure(figsize=(10.5, 8.5))
    if G.number_of_nodes()==0 or G.number_of_edges()==0:
        plt.title(title, fontproperties=FP); plt.axis("off")
        plt.savefig(outpath, dpi=320, bbox_inches="tight"); plt.close(); return

    pos=nx.spring_layout(G, seed=seed, k=0.6, weight="weight")

    freqs=np.array([G.nodes[n].get("freq",1) for n in G.nodes()])
    node_sizes=80 + 120*np.log1p(freqs)

    edge_weights=np.array([G.edges[e].get("weight",0.0) for e in G.edges()])
    edge_counts=np.array([G.edges[e].get("count",1) for e in G.edges()])
    widths=0.6 + 4.0*(edge_weights-edge_weights.min())/(edge_weights.max()-edge_weights.min()+1e-9)
    widths=widths*(0.7+0.3*np.log1p(edge_counts)/np.log1p(edge_counts.max()))

    nx.draw_networkx_edges(G,pos,width=widths,alpha=0.55)
    nx.draw_networkx_nodes(G,pos,node_size=node_sizes,alpha=0.9)
    for n,(x,y) in pos.items():
        plt.text(x,y,n,fontproperties=FP,fontsize=11,ha="center",va="center")

    plt.title(title, fontproperties=FP); plt.axis("off")
    plt.savefig(outpath, dpi=320, bbox_inches="tight"); plt.close()

# =========================
# 6) 차등 네트워크: NPMI 차이
# =========================
def diff_npmi(pair_a,pair_b,npmi_a,npmi_b,min_total=10,top_k=90):
    feats=[f for f in set(pair_a)|set(pair_b) if pair_a.get(f,0)+pair_b.get(f,0)>=min_total]
    diffs=[]
    for f in feats:
        d=npmi_a.get(f,0.0)-npmi_b.get(f,0.0)
        total=pair_a.get(f,0)+pair_b.get(f,0)
        score=d*math.log(total+1)  # 낮은 근거량 페널티
        diffs.append((f,d,score,total))
    pos=sorted([x for x in diffs if x[2]>0], key=lambda x:x[2], reverse=True)[:top_k]
    neg=sorted([x for x in diffs if x[2]<0], key=lambda x:x[2])[:top_k]
    return pos, neg

def build_diff_graph(pos_edges, base_node_freq):
    G=nx.Graph()
    for (a,b), d, score, total in pos_edges:
        G.add_node(a, freq=base_node_freq.get(a,1))
        G.add_node(b, freq=base_node_freq.get(b,1))
        G.add_edge(a,b, weight=abs(d), count=total, score=abs(score))
    return G

# =========================
# 7) 실행: 개인 vs 관청
# =========================
node_pr, pair_pr = cooc_counts(df.loc[df.group_pc=="개인","tokens"].tolist(), window=5)
node_of, pair_of = cooc_counts(df.loc[df.group_pc=="관청","tokens"].tolist(), window=5)

G_pr, npmi_pr = build_graph(node_pr,pair_pr, top_nodes=70, min_node=8, min_pair=3, top_edges=220)
G_of, npmi_of = build_graph(node_of,pair_of, top_nodes=70, min_node=12, min_pair=5, top_edges=220)

draw_network(G_pr, "개인(0 포함) 1-gram 의미연결망 (NPMI)", os.path.join(OUTDIR,"network_private.png"))
draw_network(G_of, "관청 1-gram 의미연결망 (NPMI)", os.path.join(OUTDIR,"network_official.png"))

pos_pc, neg_pc = diff_npmi(pair_pr,pair_of,npmi_pr,npmi_of,min_total=10,top_k=90)
G_priv_over = build_diff_graph(pos_pc, node_pr)
G_off_over  = build_diff_graph([(e,-d,-s,t) for (e,d,s,t) in neg_pc], node_of)
draw_network(G_priv_over, "개인에서 특히 강한 결합(개인 > 관청) 차등 의미연결망", os.path.join(OUTDIR,"diff_private_over_official.png"))
draw_network(G_off_over,  "관청에서 특히 강한 결합(관청 > 개인) 차등 의미연결망", os.path.join(OUTDIR,"diff_official_over_private.png"))

# =========================
# 8) 실행: 4개 관청 + one-vs-rest 차등
# =========================
office_rows=[]
for t in targets:
    node_t, pair_t = cooc_counts(df.loc[df.office==t,"tokens"].tolist(), window=5)
    G_t, npmi_t = build_graph_auto(node_t,pair_t, top_nodes=70)
    draw_network(G_t, f"{t} 1-gram 의미연결망 (NPMI)", os.path.join(OUTDIR,f"network_office_{t}.png"))

    node_r, pair_r = cooc_counts(df.loc[df.office.notna() & (df.office!=t),"tokens"].tolist(), window=5)
    npmi_r = edge_npmi(node_r,pair_r) if sum(node_r.values())>0 else {}
    pos_t, _ = diff_npmi(pair_t,pair_r,npmi_t,npmi_r,min_total=5,top_k=90)
    G_t_over = build_diff_graph(pos_t, node_t)
    draw_network(G_t_over, f"{t}에서 특히 강한 결합({t} > 타 3관청) 차등 의미연결망", os.path.join(OUTDIR,f"diff_office_{t}_over_rest.png"))

    for rank,(edge,d,score,total) in enumerate(pos_t[:15], start=1):
        a,b=edge
        office_rows.append([t,rank,a,b,float(d),float(score),int(total)])

pd.DataFrame(
    office_rows,
    columns=["관청","순위","노드1","노드2","NPMI차이(d)","근거보정score","총동시출현(total)"]
).to_csv(os.path.join(OUTDIR,"office_diff_top_edges.csv"), index=False, encoding="utf-8-sig")

# =========================
# 9) ZIP 생성(선택)
# =========================
zip_path=os.path.join(os.path.dirname(OUTDIR.rstrip("/")),"1gram_의미연결망_재현결과.zip")
with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(OUTDIR):
        z.write(os.path.join(OUTDIR,fn), arcname=fn)

print("DONE:", OUTDIR)
print("ZIP :", zip_path)
