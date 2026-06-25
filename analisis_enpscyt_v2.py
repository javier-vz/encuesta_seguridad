# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:13:56 2026

@author: jvera
"""

# ============================================================
# ANÁLISIS ENPSCyT 2024 - VERSIÓN 2
# Arquitectura relacional de la cultura científica en el Perú:
# información, confianza y apoyo ciudadano a la inversión pública en CTI
#
# Esta versión agrega:
# 1. Análisis piloto ponderado de la variable dependiente.
# 2. Dimensiones sustantivas predefinidas.
# 3. Red de variables como validación estructural.
# 4. Detección de comunidades con Louvain si está disponible.
# 5. Embeddings espectrales de grafos + node2vec opcional.
# 6. Índices teóricos e índices derivados de comunidades.
# 7. Modelos logísticos ponderados comparados.
# 8. Segmentación preliminar de perfiles ciudadanos.
# 9. Salidas para propuesta: tablas, figuras y resumen piloto.
# ============================================================

from pathlib import Path
from collections import defaultdict
import re
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

BASE_FILE = "Base de datos ENC_ENPSCYT 2024.csv"
DICT_FILE = "Diccionario Base de datos ENC_ENPSCYT 2024.xlsx"

base_path = BASE_DIR / BASE_FILE
dict_path = BASE_DIR / DICT_FILE

OUT_DIR = BASE_DIR / "resultados_enpscyt_v2"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. CARGA DE DATOS
# ============================================================

def read_csv_flexible(path):
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError("No se pudo leer el CSV con utf-8, utf-8-sig ni latin1.")


df = read_csv_flexible(base_path)

if dict_path.exists():
    dic = pd.read_excel(dict_path)
else:
    dic = None

print("\n==============================")
print("CARGA DE DATOS")
print("==============================")
print("Base:", df.shape)
print("Diccionario:", dic.shape if dic is not None else "No cargado")
print("Primeras columnas:")
print(df.columns[:25].tolist())


# ============================================================
# 3. FUNCIONES DE APOYO
# ============================================================

missing_values = {
    "No sabe",
    "No contesta",
    "No sabe / No contesta",
    "NS/NC",
    "Ns/Nc",
    "Ns/Nc ",
    "",
    "nan",
    "NaN"
}

ordinal_map = {
    # Satisfacción
    "Nada satisfecha/o": 1,
    "Poco satisfecha/o": 2,
    "Bastante satisfecha/o": 3,
    "Muy satisfecha/o": 4,

    # Progreso país
    "Está en retroceso": 1,
    "Está estancado": 2,
    "Está progresando": 3,

    # Escalas generales
    "Nada": 1,
    "Poco": 2,
    "Algo": 3,
    "Bastante": 4,
    "Mucho": 5,

    # Interés
    "Nada interesada/o": 1,
    "Poco interesada/o": 2,
    "Bastante interesada/o": 3,
    "Muy interesada/o": 4,

    # Información
    "Nada informada/o": 1,
    "Poco informada/o": 2,
    "Bastante informada/o": 3,
    "Muy informada/o": 4,

    # Frecuencia
    "Casi nunca o nunca": 1,
    "De vez en cuando": 2,
    "Con frecuencia": 3,

    # Sí / No
    "No": 0,
    "Sí": 1,
    "Si": 1,

    # Acuerdo
    "Muy en desacuerdo": 1,
    "En desacuerdo": 2,
    "De acuerdo": 3,
    "De Acuerdo": 3,
    "Muy de acuerdo": 4,

    # Inversión
    "Tendría que disminuir": 1,
    "Tendría que permanecer igual": 2,
    "Tendría que aumentar": 3,

    # Desarrollo CyT
    "Atrasado": 1,
    "En un lugar intermedio": 2,
    "Adelantado": 3,

    # Condiciones para investigar
    "Muy malas": 1,
    "Malas": 2,
    "Buenas": 3,
    "Muy buenas": 4,

    # Financiamiento
    "Muy insuficiente": 1,
    "Insuficiente": 2,
    "Razonablemente suficiente": 3,
    "Muy suficiente": 4,

    # Profesión científica
    "Nada atractiva": 1,
    "Poco atractiva": 2,
    "Bastante atractiva": 3,
    "Muy atractiva": 4,

    "Nada gratificante en lo personal": 1,
    "Poco gratificante en lo personal": 2,
    "Bastante gratificante en lo personal": 3,
    "Muy gratificante en lo personal": 4,

    "Muy mal remunerada": 1,
    "Mal remunerada": 2,
    "Bien remunerada": 3,
    "Muy bien remunerada": 4,

    "Nada prestigiosa": 1,
    "Poco prestigiosa": 2,
    "Bastante prestigiosa": 3,
    "Muy prestigiosa": 4,

    # Futuro de la investigación
    "Nada destacado": 1,
    "Poco destacado": 2,
    "Bastante destacado": 3,
    "Muy destacado": 4,

    # IA
    "Muy negativos": 1,
    "Negativos": 2,
    "Positivos": 3,
    "Muy positivos": 4,
}


def recode_ordinal(series: pd.Series) -> pd.Series:
    """
    Convierte respuestas textuales a escala numérica.
    También captura respuestas tipo:
    '1 Nada de acuerdo', '10 Totalmente de acuerdo', etc.
    """

    def one_value(x):
        if pd.isna(x):
            return np.nan

        x = str(x).strip()

        if x in missing_values:
            return np.nan

        if x in ordinal_map:
            return ordinal_map[x]

        m = re.match(r"^(\d+)", x)
        if m:
            return float(m.group(1))

        return np.nan

    return series.map(one_value).astype(float)


def select_cols(dataframe, patterns, exclude=("OTRO", "Otro", "otro")):
    """
    Selecciona columnas por prefijo o nombre exacto.
    """

    cols = []

    for pat in patterns:
        if pat.endswith("_"):
            selected = [
                c for c in dataframe.columns
                if c.startswith(pat) and not any(e in c for e in exclude)
            ]
            cols.extend(selected)
        else:
            if pat in dataframe.columns:
                cols.append(pat)

    return list(dict.fromkeys(cols))


def normalize_dic_columns(dic_df):
    """
    Intenta encontrar columnas de variable y descripción en el diccionario.
    """

    if dic_df is None:
        return None, None

    var_col = None
    desc_col = None

    for c in dic_df.columns:
        c_norm = str(c).strip().lower()
        if c_norm in ["variable", "variables", "var"]:
            var_col = c

    for c in dic_df.columns:
        c_norm = str(c).strip().lower()
        if c_norm in ["descripción", "descripcion", "description", "pregunta"]:
            desc_col = c

    return var_col, desc_col


DIC_VAR_COL, DIC_DESC_COL = normalize_dic_columns(dic)


def var_label(var, maxlen=120):
    """
    Devuelve etiqueta legible desde el diccionario.
    """

    if dic is None:
        return var

    if DIC_VAR_COL is None or DIC_DESC_COL is None:
        return var

    row = dic.loc[dic[DIC_VAR_COL].astype(str) == str(var), DIC_DESC_COL]

    if len(row) == 0:
        return var

    txt = str(row.iloc[0])
    txt = re.sub(r"\s+", " ", txt).strip()

    if len(txt) > maxlen:
        txt = txt[:maxlen] + "..."

    return txt


def weighted_mean(series, weights):
    mask = series.notna() & weights.notna() & (weights > 0)

    if mask.sum() == 0:
        return np.nan

    x = series[mask].astype(float)
    ww = weights[mask].astype(float)

    return np.sum(x * ww) / np.sum(ww)


def weighted_mean_std(series, weights):
    mask = series.notna() & weights.notna() & (weights > 0)

    x = series[mask].astype(float)
    ww = weights[mask].astype(float)

    if len(x) == 0:
        return np.nan, np.nan

    ww = ww / ww.sum()

    mean = np.sum(ww * x)
    var = np.sum(ww * (x - mean) ** 2)

    return mean, np.sqrt(var)


def weighted_corr(x, y, w, min_n=500):
    """
    Correlación ponderada entre dos variables.
    Aplicada sobre rankings aproxima Spearman ponderada.
    """

    mask = x.notna() & y.notna() & w.notna() & (w > 0)

    if mask.sum() < min_n:
        return np.nan

    xx = x[mask].to_numpy(dtype=float)
    yy = y[mask].to_numpy(dtype=float)
    ww = w[mask].to_numpy(dtype=float)

    ww = ww / ww.sum()

    mx = np.sum(ww * xx)
    my = np.sum(ww * yy)

    vx = np.sum(ww * (xx - mx) ** 2)
    vy = np.sum(ww * (yy - my) ** 2)

    if vx <= 0 or vy <= 0:
        return np.nan

    cov = np.sum(ww * (xx - mx) * (yy - my))

    return cov / np.sqrt(vx * vy)


def cronbach_alpha(dataframe):
    """
    Alfa de Cronbach simple sobre variables numéricas.
    """
    X_local = dataframe.dropna()

    if X_local.shape[1] < 2 or X_local.shape[0] < 10:
        return np.nan

    item_vars = X_local.var(axis=0, ddof=1)
    total_var = X_local.sum(axis=1).var(ddof=1)
    k = X_local.shape[1]

    if total_var <= 0:
        return np.nan

    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def fit_logit_weighted(data, y_col, predictors, weight_col="w_norm"):
    """
    Ajusta GLM binomial ponderado.
    """

    X_tmp = pd.get_dummies(data[predictors], drop_first=True)
    X_tmp = sm.add_constant(X_tmp).astype(float)

    y_tmp = data[y_col].astype(float)
    w_tmp = data[weight_col].astype(float)

    mod = sm.GLM(
        y_tmp,
        X_tmp,
        family=sm.families.Binomial(),
        freq_weights=w_tmp
    )

    res = mod.fit(cov_type="HC1")

    return res, X_tmp.columns.tolist()


# ============================================================
# 4. FACTOR DE EXPANSIÓN
# ============================================================

if "Fexp" not in df.columns:
    raise ValueError("No se encontró la columna Fexp en la base.")

w = pd.to_numeric(df["Fexp"], errors="coerce").astype(float)

print("\n==============================")
print("FACTOR DE EXPANSIÓN")
print("==============================")
print(w.describe())


# ============================================================
# 5. VARIABLE DEPENDIENTE Y ANÁLISIS PILOTO
# ============================================================

if "P20" not in df.columns:
    raise ValueError("No se encontró P20 en la base.")

y = np.where(
    df["P20"].eq("Tendría que aumentar"),
    1,
    np.where(
        df["P20"].isin([
            "Tendría que permanecer igual",
            "Tendría que disminuir"
        ]),
        0,
        np.nan
    )
)

df["y_apoya_aumento"] = y
df["w_norm"] = w / w.mean()

dep_summary = pd.DataFrame({
    "categoria": ["Apoya aumento", "No apoya aumento"],
    "proporcion_ponderada": [
        weighted_mean(pd.Series(df["y_apoya_aumento"].eq(1).astype(float)), w),
        weighted_mean(pd.Series(df["y_apoya_aumento"].eq(0).astype(float)), w)
    ]
})

dep_summary.to_csv(OUT_DIR / "piloto_variable_dependiente.csv", index=False)

print("\n==============================")
print("VARIABLE DEPENDIENTE")
print("==============================")
print(df["P20"].value_counts(dropna=False))
print("\nResumen ponderado:")
print(dep_summary)


# ============================================================
# 6. DIMENSIONES SUSTANTIVAS PREDEFINIDAS
# ============================================================
# Estas dimensiones son una primera aproximación metodológica.
# Luego deben ajustarse revisando el diccionario y los resultados.

dimension_patterns = {
    "capital_informativo": [
        "P4_", "P5_", "P6_"
    ],
    "participacion_cultural_cientifica": [
        "P7_"
    ],
    "confianza_valoracion_cientificos": [
        "P9_", "P11", "P12", "P13", "P14", "P15"
    ],
    "institucionalidad_cti": [
        "P18", "P19", "P22", "P24_", "P25"
    ],
    "optimismo_tecnocientifico": [
        "P26_", "P30_", "P35", "P36"
    ],
    "actitudes_ia_riesgos_beneficios": [
        "P37_"
    ],
    "percepcion_capacidades_nacionales": [
        "P3_"
    ],
    "contexto_pais": [
        "P1_", "P2"
    ],
}

# P20 no se usa como predictor ni como parte de índices.
all_dim_cols = []

for dim, pats in dimension_patterns.items():
    cols_dim = select_cols(df, pats)
    cols_dim = [c for c in cols_dim if c != "P20"]
    all_dim_cols.extend(cols_dim)

vars_all = list(dict.fromkeys(all_dim_cols))

print("\n==============================")
print("DIMENSIONES PREDEFINIDAS")
print("==============================")

for dim, pats in dimension_patterns.items():
    cols_dim = [c for c in select_cols(df, pats) if c != "P20"]
    print(f"{dim}: {len(cols_dim)} variables")


# ============================================================
# 7. MATRIZ NUMÉRICA RECODIFICADA
# ============================================================

X = pd.DataFrame({
    col: recode_ordinal(df[col])
    for col in vars_all
})

# Filtrado de calidad
X = X.loc[:, X.notna().mean().ge(0.50)]
X = X.loc[:, X.nunique(dropna=True).gt(1)]

print("\n==============================")
print("MATRIZ NUMÉRICA")
print("==============================")
print("X:", X.shape)

# Guardar cobertura de variables
coverage = pd.DataFrame({
    "variable": X.columns,
    "label": [var_label(c) for c in X.columns],
    "prop_no_missing": [X[c].notna().mean() for c in X.columns],
    "n_unique": [X[c].nunique(dropna=True) for c in X.columns],
})

coverage.to_csv(OUT_DIR / "piloto_cobertura_variables.csv", index=False)


# ============================================================
# 8. ÍNDICES TEÓRICOS PRELIMINARES
# ============================================================

Z_dict = {}

for c in X.columns:
    mu, sd = weighted_mean_std(X[c], w)
    if pd.notna(sd) and sd > 0:
        Z_dict[c] = (X[c] - mu) / sd

Z = pd.DataFrame(Z_dict, index=df.index)

theory_scores = pd.DataFrame(index=df.index)
theory_index_rows = []

for dim, pats in dimension_patterns.items():
    cols_dim = [c for c in select_cols(df, pats) if c in Z.columns and c != "P20"]

    if len(cols_dim) >= 2:
        theory_scores[f"idx_{dim}"] = Z[cols_dim].mean(axis=1)

        alpha = cronbach_alpha(X[cols_dim])

        for c in cols_dim:
            theory_index_rows.append({
                "dimension": dim,
                "variable": c,
                "label": var_label(c),
                "alpha_dimension": alpha
            })

theory_index_df = pd.DataFrame(theory_index_rows)
theory_index_df.to_csv(OUT_DIR / "indices_teoricos_composicion.csv", index=False)

print("\n==============================")
print("ÍNDICES TEÓRICOS")
print("==============================")
print(theory_scores.shape)
print(theory_index_df.groupby("dimension").size())


# ============================================================
# 9. RED DE VARIABLES: CORRELACIONES PONDERADAS TIPO SPEARMAN
# ============================================================

print("\n==============================")
print("RED DE VARIABLES")
print("==============================")

X_rank = X.rank(axis=0, method="average", na_option="keep")

cols = X_rank.columns.tolist()
corr = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

for i, c1 in enumerate(cols):
    for j in range(i + 1, len(cols)):
        c2 = cols[j]
        r = weighted_corr(X_rank[c1], X_rank[c2], w, min_n=500)
        corr.loc[c1, c2] = r
        corr.loc[c2, c1] = r

corr.to_csv(OUT_DIR / "matriz_correlaciones_ponderadas.csv")


threshold = 0.30

G = nx.Graph()

for c in corr.columns:
    G.add_node(c, label=var_label(c))

for i, c1 in enumerate(corr.columns):
    for j in range(i + 1, len(corr.columns)):
        c2 = corr.columns[j]
        r = corr.loc[c1, c2]

        if pd.notna(r) and abs(r) >= threshold:
            G.add_edge(
                c1,
                c2,
                weight=abs(r),
                rho=r,
                sign=1 if r > 0 else -1
            )

print("Nodos:", G.number_of_nodes())
print("Aristas:", G.number_of_edges())
print("Componentes:", nx.number_connected_components(G))

if G.number_of_nodes() > 0:
    print("Componente mayor:", len(max(nx.connected_components(G), key=len)))


# ============================================================
# 10. DETECCIÓN DE COMUNIDADES
# ============================================================

def detect_communities(G):
    """
    Usa Louvain si está disponible en NetworkX.
    Si no, usa greedy modularity.
    """

    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes()], "none"

    try:
        comms = nx.algorithms.community.louvain_communities(
            G,
            weight="weight",
            seed=42,
            resolution=1.0
        )
        method = "louvain"
    except Exception:
        comms = nx.algorithms.community.greedy_modularity_communities(
            G,
            weight="weight"
        )
        method = "greedy_modularity"

    comms = sorted([set(c) for c in comms], key=len, reverse=True)

    return comms, method


communities, comm_method = detect_communities(G)

node_to_comm = {}

for k, comm in enumerate(communities, start=1):
    for node in comm:
        node_to_comm[node] = k

try:
    modularity = nx.algorithms.community.modularity(
        G,
        communities,
        weight="weight"
    )
except Exception:
    modularity = np.nan

print("\nMétodo comunidades:", comm_method)
print("Modularidad:", modularity)

comm_rows = []

for k, comm in enumerate(communities, start=1):
    sub = G.subgraph(comm)
    top = sorted(
        sub.degree(weight="weight"),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(f"\nComunidad {k} | tamaño: {len(comm)}")
    for var, deg in top:
        print(f"{var:8s} | {deg:.2f} | {var_label(var)}")

    for v in sorted(comm):
        comm_rows.append({
            "community_id": k,
            "variable": v,
            "label": var_label(v),
            "weighted_degree_in_comm": sub.degree(v, weight="weight")
        })

communities_df = pd.DataFrame(comm_rows)
communities_df.to_csv(OUT_DIR / "comunidades_variables.csv", index=False)


# ============================================================
# 11. ROBUSTEZ DE COMUNIDADES POR UMBRAL
# ============================================================

robust_rows = []

for th in [0.25, 0.30, 0.35, 0.40]:
    G_th = nx.Graph()
    for c in corr.columns:
        G_th.add_node(c)

    for i, c1 in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            c2 = corr.columns[j]
            r = corr.loc[c1, c2]

            if pd.notna(r) and abs(r) >= th:
                G_th.add_edge(c1, c2, weight=abs(r), rho=r)

    comms_th, method_th = detect_communities(G_th)

    try:
        mod_th = nx.algorithms.community.modularity(
            G_th,
            comms_th,
            weight="weight"
        )
    except Exception:
        mod_th = np.nan

    robust_rows.append({
        "threshold": th,
        "nodes": G_th.number_of_nodes(),
        "edges": G_th.number_of_edges(),
        "components": nx.number_connected_components(G_th),
        "n_communities": len(comms_th),
        "largest_component": len(max(nx.connected_components(G_th), key=len)) if G_th.number_of_nodes() else 0,
        "modularity": mod_th,
        "method": method_th
    })

robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(OUT_DIR / "robustez_red_umbral.csv", index=False)

print("\n==============================")
print("ROBUSTEZ DE RED")
print("==============================")
print(robust_df)


# ============================================================
# 12. ARISTAS Y MÉTRICAS DE NODOS
# ============================================================

edges = []

for u, v, data in G.edges(data=True):
    edges.append({
        "source": u,
        "target": v,
        "rho": data["rho"],
        "abs_rho": abs(data["rho"]),
        "sign": data["sign"],
        "source_community": node_to_comm.get(u),
        "target_community": node_to_comm.get(v),
        "source_label": var_label(u),
        "target_label": var_label(v),
    })

edges_df = pd.DataFrame(edges)

if not edges_df.empty:
    edges_df = edges_df.sort_values("abs_rho", ascending=False)

edges_df.to_csv(OUT_DIR / "red_aristas_variables.csv", index=False)

degree_w = dict(G.degree(weight="weight"))

try:
    betweenness_w = nx.betweenness_centrality(
        G,
        weight="weight",
        normalized=True
    )
except Exception:
    betweenness_w = {n: np.nan for n in G.nodes()}

nodes_df = pd.DataFrame({
    "variable": list(G.nodes()),
    "label": [var_label(n) for n in G.nodes()],
    "community_id": [node_to_comm.get(n) for n in G.nodes()],
    "weighted_degree": [degree_w.get(n, 0) for n in G.nodes()],
    "betweenness": [betweenness_w.get(n, np.nan) for n in G.nodes()],
})

nodes_df = nodes_df.sort_values("weighted_degree", ascending=False)
nodes_df.to_csv(OUT_DIR / "red_metricas_nodos.csv", index=False)


# ============================================================
# 13. EMBEDDINGS ESPECTRALES DE GRAFO
# ============================================================

def spectral_graph_embeddings(G, dim=4):
    """
    Embedding espectral simple usando el Laplaciano normalizado.
    Devuelve coordenadas de nodos en dimensiones latentes.
    """

    if G.number_of_nodes() < 3:
        return pd.DataFrame()

    nodes = list(G.nodes())

    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    degrees = A.sum(axis=1)

    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(degrees)

    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(len(nodes)) - D_inv_sqrt @ A @ D_inv_sqrt

    vals, vecs = np.linalg.eigh(L_norm)

    # Saltamos el primer autovector trivial.
    max_dim = min(dim, vecs.shape[1] - 1)
    coords = vecs[:, 1:1 + max_dim]

    emb = pd.DataFrame(coords, columns=[f"spec_emb_{i+1}" for i in range(max_dim)])
    emb["variable"] = nodes
    emb["label"] = [var_label(n) for n in nodes]
    emb["community_id"] = [node_to_comm.get(n) for n in nodes]

    return emb


spectral_emb_df = spectral_graph_embeddings(G, dim=4)
spectral_emb_df.to_csv(OUT_DIR / "graph_embeddings_espectrales.csv", index=False)

print("\nEmbeddings espectrales:", spectral_emb_df.shape)


# ============================================================
# 14. NODE2VEC OPCIONAL
# ============================================================
# Corre solo si el paquete node2vec está instalado.
# No es obligatorio para la propuesta; queda como componente opcional.

try:
    from node2vec import Node2Vec

    if G.number_of_edges() > 0:
        node2vec = Node2Vec(
            G,
            dimensions=8,
            walk_length=12,
            num_walks=80,
            workers=1,
            weight_key="weight",
            seed=42
        )

        n2v_model = node2vec.fit(
            window=5,
            min_count=1,
            batch_words=4
        )

        rows = []

        for n in G.nodes():
            vec = n2v_model.wv[str(n)] if str(n) in n2v_model.wv else n2v_model.wv[n]
            row = {
                "variable": n,
                "label": var_label(n),
                "community_id": node_to_comm.get(n)
            }
            for i, val in enumerate(vec):
                row[f"node2vec_{i+1}"] = val
            rows.append(row)

        node2vec_df = pd.DataFrame(rows)
        node2vec_df.to_csv(OUT_DIR / "graph_embeddings_node2vec.csv", index=False)
        print("Node2vec calculado:", node2vec_df.shape)

except Exception as e:
    print("Node2vec no ejecutado. Motivo:", str(e))


# ============================================================
# 15. ÍNDICES DERIVADOS DE COMUNIDADES DE RED
# ============================================================

community_scores = pd.DataFrame(index=df.index)
community_vars = {}

for k, comm in enumerate(communities, start=1):
    vars_k = [v for v in comm if v in Z.columns]

    if len(vars_k) >= 3:
        name = f"red_com_{k:02d}"
        community_scores[name] = Z[vars_k].mean(axis=1)
        community_vars[name] = vars_k

index_rows = []

for name, vars_k in community_vars.items():
    alpha = cronbach_alpha(X[vars_k])

    for v in vars_k:
        index_rows.append({
            "index": name,
            "variable": v,
            "label": var_label(v),
            "alpha_index": alpha
        })

community_indices_df = pd.DataFrame(index_rows)
community_indices_df.to_csv(OUT_DIR / "indices_red_composicion.csv", index=False)

print("\nÍndices de red:", community_scores.shape)


# ============================================================
# 16. MODELOS LOGÍSTICOS PONDERADOS
# ============================================================

control_candidates = [
    "P41",          # sexo
    "P42",          # edad
    "p43_grupo",    # nivel educativo
    "mod0_5",       # NSE
    "mod0_1_dep"    # departamento
]

controls = [c for c in control_candidates if c in df.columns]

model_df = pd.DataFrame(index=df.index)
model_df["y_apoya_aumento"] = df["y_apoya_aumento"]
model_df["w_norm"] = df["w_norm"]

for c in controls:
    model_df[c] = df[c]

# Agregar índices teóricos
for c in theory_scores.columns:
    model_df[c] = theory_scores[c]

# Agregar índices de red
for c in community_scores.columns:
    model_df[c] = community_scores[c]

theory_cols = theory_scores.columns.tolist()
red_cols = community_scores.columns.tolist()

base_model_cols = ["y_apoya_aumento", "w_norm"] + controls
theory_model_cols = base_model_cols + theory_cols
red_model_cols = base_model_cols + red_cols

# Usamos una misma muestra para comparar modelos de manera justa.
all_needed = ["y_apoya_aumento", "w_norm"] + controls + theory_cols + red_cols
m_all = model_df[all_needed].dropna().copy()

print("\n==============================")
print("MODELOS")
print("==============================")
print("Casos completos:", len(m_all))

model_results = []

# Modelo 1: sociodemográfico
res_base, cols_base = fit_logit_weighted(
    m_all,
    "y_apoya_aumento",
    controls
)

model_results.append({
    "modelo": "M1_base_sociodemografico",
    "AIC": res_base.aic,
    "LogLik": res_base.llf,
    "Df_model": res_base.df_model,
    "N": len(m_all)
})

# Modelo 2: controles + índices teóricos
res_theory, cols_theory = fit_logit_weighted(
    m_all,
    "y_apoya_aumento",
    controls + theory_cols
)

model_results.append({
    "modelo": "M2_indices_teoricos",
    "AIC": res_theory.aic,
    "LogLik": res_theory.llf,
    "Df_model": res_theory.df_model,
    "N": len(m_all)
})

# Modelo 3: controles + índices derivados de red
res_red, cols_red = fit_logit_weighted(
    m_all,
    "y_apoya_aumento",
    controls + red_cols
)

model_results.append({
    "modelo": "M3_indices_red",
    "AIC": res_red.aic,
    "LogLik": res_red.llf,
    "Df_model": res_red.df_model,
    "N": len(m_all)
})

# Modelo 4: controles + índices teóricos + red
# Puede tener colinealidad, pero sirve como exploración.
res_combined, cols_combined = fit_logit_weighted(
    m_all,
    "y_apoya_aumento",
    controls + theory_cols + red_cols
)

model_results.append({
    "modelo": "M4_teoricos_y_red",
    "AIC": res_combined.aic,
    "LogLik": res_combined.llf,
    "Df_model": res_combined.df_model,
    "N": len(m_all)
})

comparison = pd.DataFrame(model_results)
comparison["delta_AIC_vs_base"] = comparison["AIC"] - comparison.loc[0, "AIC"]
comparison.to_csv(OUT_DIR / "comparacion_modelos_v2.csv", index=False)

print(comparison)


def odds_ratios_table(res, selected_cols, out_name):
    params = res.params
    ci = res.conf_int()
    pvals = res.pvalues

    rows = []

    for col in selected_cols:
        if col in params.index:
            rows.append({
                "variable": col,
                "OR": np.exp(params[col]),
                "IC95_inf": np.exp(ci.loc[col, 0]),
                "IC95_sup": np.exp(ci.loc[col, 1]),
                "p_value": pvals[col]
            })

    tab = pd.DataFrame(rows).sort_values("OR", ascending=False)
    tab.to_csv(OUT_DIR / out_name, index=False)

    return tab


or_theory = odds_ratios_table(
    res_theory,
    theory_cols,
    "modelo_or_indices_teoricos.csv"
)

or_red = odds_ratios_table(
    res_red,
    red_cols,
    "modelo_or_indices_red.csv"
)

print("\nOR índices teóricos")
print(or_theory)

print("\nOR índices de red")
print(or_red)


# ============================================================
# 17. EFECTOS MARGINALES PROMEDIO
# ============================================================

try:
    marg_theory = res_theory.get_margeff(at="overall").summary_frame()
    marg_theory.to_csv(OUT_DIR / "efectos_marginales_modelo_teorico.csv")
except Exception as e:
    print("No se pudieron calcular efectos marginales modelo teórico:", e)

try:
    marg_red = res_red.get_margeff(at="overall").summary_frame()
    marg_red.to_csv(OUT_DIR / "efectos_marginales_modelo_red.csv")
except Exception as e:
    print("No se pudieron calcular efectos marginales modelo red:", e)


# ============================================================
# 18. PERFILES CIUDADANOS
# ============================================================

print("\n==============================")
print("PERFILES CIUDADANOS")
print("==============================")

# Usamos índices teóricos para perfiles porque son más interpretables.
profile_cols = theory_cols.copy()

profiles_df = model_df[["y_apoya_aumento", "w_norm"] + controls + profile_cols].dropna().copy()

# K-means con sklearn si está disponible.
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X_prof = profiles_df[profile_cols].copy()
    scaler = StandardScaler()
    X_prof_scaled = scaler.fit_transform(X_prof)

    profile_results = []

    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_prof_scaled)

        sil = silhouette_score(X_prof_scaled, labels)

        profile_results.append({
            "k": k,
            "silhouette": sil,
            "inertia": km.inertia_
        })

    profile_results_df = pd.DataFrame(profile_results)
    profile_results_df.to_csv(OUT_DIR / "perfiles_kmeans_comparacion.csv", index=False)

    best_k = int(profile_results_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

    km = KMeans(n_clusters=best_k, random_state=42, n_init=50)
    profiles_df["perfil"] = km.fit_predict(X_prof_scaled) + 1

    profile_summary = []

    for p in sorted(profiles_df["perfil"].unique()):
        sub = profiles_df[profiles_df["perfil"] == p]

        row = {
            "perfil": p,
            "n": len(sub),
            "prop_muestra": len(sub) / len(profiles_df),
            "apoyo_inversion_ponderado": weighted_mean(sub["y_apoya_aumento"], sub["w_norm"])
        }

        for c in profile_cols:
            row[f"mean_{c}"] = weighted_mean(sub[c], sub["w_norm"])

        profile_summary.append(row)

    profile_summary_df = pd.DataFrame(profile_summary)
    profile_summary_df.to_csv(OUT_DIR / "perfiles_ciudadanos_resumen.csv", index=False)

    profiles_df.to_csv(OUT_DIR / "base_perfiles_ciudadanos.csv", index=False)

    print("Mejor k:", best_k)
    print(profile_results_df)
    print(profile_summary_df)

except Exception as e:
    print("No se ejecutó clustering. Motivo:", e)


# ============================================================
# 19. VISUALIZACIONES
# ============================================================

def save_bar_or(or_df, title, filename):
    if or_df.empty:
        return

    plot_df = or_df.copy().sort_values("OR", ascending=True)

    plt.figure(figsize=(8, max(4, 0.45 * len(plot_df))))
    plt.errorbar(
        plot_df["OR"],
        plot_df["variable"],
        xerr=[
            plot_df["OR"] - plot_df["IC95_inf"],
            plot_df["IC95_sup"] - plot_df["OR"]
        ],
        fmt="o",
        capsize=3
    )
    plt.axvline(1, linestyle="--", linewidth=1)
    plt.xlabel("Odds ratio")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


save_bar_or(
    or_theory,
    "Asociación entre índices teóricos y apoyo a aumentar inversión en CTI",
    "fig_or_indices_teoricos.png"
)

save_bar_or(
    or_red,
    "Asociación entre índices de red y apoyo a aumentar inversión en CTI",
    "fig_or_indices_red.png"
)


# Red de variables compacta
nodes_viz = [n for n in G.nodes() if G.degree(n) > 0]
G_viz = G.subgraph(nodes_viz).copy()

if G_viz.number_of_edges() > 0:
    G_layout = G_viz.copy()

    components = sorted(nx.connected_components(G_layout), key=len, reverse=True)
    component_hubs = []

    for comp in components:
        sub = G_layout.subgraph(comp)
        hub = max(sub.nodes(), key=lambda n: sub.degree(n, weight="weight"))
        component_hubs.append(hub)

    if len(component_hubs) > 1:
        main_hub = component_hubs[0]
        for hub in component_hubs[1:]:
            G_layout.add_edge(main_hub, hub, weight=0.18, phantom=True)

    pos = nx.spring_layout(
        G_layout,
        seed=42,
        weight="weight",
        k=0.28,
        iterations=1500,
        scale=1.0
    )

    pos = {n: pos[n] for n in G_viz.nodes()}

    cmap = plt.colormaps.get_cmap("tab20")

    node_colors = [
        cmap((node_to_comm.get(n, 0) % 20) / 20)
        for n in G_viz.nodes()
    ]

    weighted_degrees = dict(G_viz.degree(weight="weight"))

    node_sizes = [
        90 + 230 * weighted_degrees.get(n, 0)
        for n in G_viz.nodes()
    ]

    labels = {}

    top_per_comm = 2

    for comm_id in sorted(set(node_to_comm.values())):
        nodes_comm = [
            n for n in G_viz.nodes()
            if node_to_comm.get(n) == comm_id
        ]

        nodes_comm_sorted = sorted(
            nodes_comm,
            key=lambda n: weighted_degrees.get(n, 0),
            reverse=True
        )

        for n in nodes_comm_sorted[:top_per_comm]:
            labels[n] = n

    plt.figure(figsize=(12, 9))

    nx.draw_networkx_edges(
        G_viz,
        pos,
        width=[0.35 + 3.2 * G_viz[u][v]["weight"] for u, v in G_viz.edges()],
        alpha=0.35,
        edge_color="gray"
    )

    nx.draw_networkx_nodes(
        G_viz,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.6,
        edgecolors="white",
        alpha=0.92
    )

    nx.draw_networkx_labels(
        G_viz,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold"
    )

    plt.title(
        "Red de asociaciones entre variables de percepción pública de la ciencia",
        fontsize=14,
        pad=14
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(OUT_DIR / "fig_red_variables_v2.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_red_variables_v2.svg", bbox_inches="tight")
    plt.close()


# ============================================================
# 20. RESUMEN PILOTO PARA PROPUESTA
# ============================================================

resumen_md = OUT_DIR / "resumen_piloto_para_propuesta.md"

with open(resumen_md, "w", encoding="utf-8") as f:
    f.write("# Resumen piloto para propuesta ENPSCyT 2024\n\n")

    f.write("## Base\n\n")
    f.write(f"- Filas: {df.shape[0]}\n")
    f.write(f"- Columnas: {df.shape[1]}\n")
    f.write(f"- Variables recodificadas para análisis relacional: {X.shape[1]}\n\n")

    f.write("## Variable dependiente\n\n")
    for _, row in dep_summary.iterrows():
        f.write(
            f"- {row['categoria']}: "
            f"{100 * row['proporcion_ponderada']:.2f}% ponderado\n"
        )

    f.write("\n## Red de variables\n\n")
    f.write(f"- Umbral de asociación: |rho| >= {threshold}\n")
    f.write(f"- Nodos: {G.number_of_nodes()}\n")
    f.write(f"- Aristas: {G.number_of_edges()}\n")
    f.write(f"- Componentes: {nx.number_connected_components(G)}\n")
    f.write(f"- Método de comunidades: {comm_method}\n")
    f.write(f"- Modularidad: {modularity:.3f}\n\n")

    f.write("## Comparación de modelos\n\n")
    f.write(comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Principales OR - índices teóricos\n\n")
    if not or_theory.empty:
        f.write(or_theory.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Principales OR - índices derivados de red\n\n")
    if not or_red.empty:
        f.write(or_red.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Archivos producidos\n\n")
    f.write("- piloto_variable_dependiente.csv\n")
    f.write("- piloto_cobertura_variables.csv\n")
    f.write("- indices_teoricos_composicion.csv\n")
    f.write("- matriz_correlaciones_ponderadas.csv\n")
    f.write("- comunidades_variables.csv\n")
    f.write("- robustez_red_umbral.csv\n")
    f.write("- graph_embeddings_espectrales.csv\n")
    f.write("- modelo_or_indices_teoricos.csv\n")
    f.write("- modelo_or_indices_red.csv\n")
    f.write("- comparacion_modelos_v2.csv\n")
    f.write("- perfiles_ciudadanos_resumen.csv, si sklearn está instalado\n")
    f.write("- fig_or_indices_teoricos.png\n")
    f.write("- fig_or_indices_red.png\n")
    f.write("- fig_red_variables_v2.png\n")

print("\n==============================")
print("RESUMEN FINAL")
print("==============================")
print("Archivos guardados en:", OUT_DIR)
print("Resumen piloto:", resumen_md)

print("\nSugerencia:")
print("Revisar resumen_piloto_para_propuesta.md para extraer resultados preliminares.")