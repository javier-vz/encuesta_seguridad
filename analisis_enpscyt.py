# ============================================================
# ANÁLISIS ENPSCYT 2024
# Arquitectura relacional de la cultura científica en el Perú
# Redes de información, actitudes tecnológicas y apoyo ciudadano
# a la inversión pública en ciencia y tecnología
# ============================================================

from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import matplotlib.pyplot as plt


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

OUT_DIR = BASE_DIR / "resultados_enpscyt"
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

    raise UnicodeDecodeError(
        "No se pudo leer el CSV con utf-8, utf-8-sig ni latin1."
    )


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

        # Captura respuestas que empiezan con número
        m = re.match(r"^(\d+)", x)
        if m:
            return float(m.group(1))

        return np.nan

    return series.map(one_value).astype(float)


def select_cols(dataframe, patterns, exclude=("OTRO", "Otro", "otro")):
    """
    Selecciona columnas por prefijo o nombre exacto.
    'P4_' selecciona P4_1, P4_2, etc.
    'P20' selecciona solo P20.
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
    Intenta encontrar las columnas de variable y descripción del diccionario.
    """

    if dic_df is None:
        return None, None

    cols = list(dic_df.columns)

    var_col = None
    desc_col = None

    for c in cols:
        c_norm = str(c).strip().lower()
        if c_norm in ["variable", "variables", "var"]:
            var_col = c

    for c in cols:
        c_norm = str(c).strip().lower()
        if c_norm in ["descripción", "descripcion", "description", "pregunta"]:
            desc_col = c

    return var_col, desc_col


DIC_VAR_COL, DIC_DESC_COL = normalize_dic_columns(dic)


def var_label(var, maxlen=100):
    """
    Devuelve una etiqueta legible desde el diccionario.
    Si no hay diccionario, devuelve el nombre de la variable.
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


def weighted_corr(x, y, w, min_n=500):
    """
    Correlación ponderada entre dos variables.
    Como se aplica sobre rankings, funciona como aproximación
    a una Spearman ponderada.
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


def weighted_mean_std(series, weights):
    """
    Media y desviación estándar ponderadas.
    """

    mask = series.notna() & weights.notna() & (weights > 0)

    x = series[mask].astype(float)
    ww = weights[mask].astype(float)
    ww = ww / ww.sum()

    mean = np.sum(ww * x)
    var = np.sum(ww * (x - mean) ** 2)

    return mean, np.sqrt(var)


# ============================================================
# 4. SELECCIÓN DE VARIABLES PARA LA RED
#    IMPORTANTE: P20 NO ENTRA EN LA RED
#    porque será variable dependiente del modelo.
# ============================================================

patterns = [
    # Contexto país
    "P1_", "P2", "P3_",

    # Interés, información y consumo cultural
    "P4_", "P5_", "P6_", "P7_",

    # Imagen de profesiones y científicos
    "P9_", "P11", "P12", "P13", "P14", "P15",

    # Institucionalidad e inversión
    # P20 se excluye de la red.
    "P18", "P19", "P22", "P24_", "P25",

    # Actitudes hacia ciencia y tecnología
    "P26_", "P30_", "P31",

    # Inteligencia artificial
    "P35", "P36", "P37_",
]

vars_red = select_cols(df, patterns)

if "P20" in vars_red:
    vars_red.remove("P20")

print("\n==============================")
print("VARIABLES PARA LA RED")
print("==============================")
print("Variables candidatas:", len(vars_red))
print(vars_red)


# ============================================================
# 5. MATRIZ NUMÉRICA RECODIFICADA
# ============================================================

X = pd.DataFrame({
    col: recode_ordinal(df[col])
    for col in vars_red
})

# Mantener variables con suficiente información
X = X.loc[:, X.notna().mean().ge(0.50)]

# Mantener variables con variación real
X = X.loc[:, X.nunique(dropna=True).gt(1)]

print("\n==============================")
print("MATRIZ NUMÉRICA")
print("==============================")
print("X:", X.shape)


# ============================================================
# 6. FACTOR DE EXPANSIÓN
# ============================================================

if "Fexp" not in df.columns:
    raise ValueError("No se encontró la columna Fexp en la base.")

w = pd.to_numeric(df["Fexp"], errors="coerce").astype(float)

print("\nFactor de expansión:")
print(w.describe())


# ============================================================
# 7. MATRIZ DE CORRELACIÓN PONDERADA TIPO SPEARMAN
# ============================================================

print("\n==============================")
print("CORRELACIONES PONDERADAS")
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

print("Matriz de correlaciones guardada.")


# ============================================================
# 8. CONSTRUCCIÓN DE RED DE VARIABLES
# ============================================================

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

print("\n==============================")
print("RED DE VARIABLES")
print("==============================")
print("Nodos:", G.number_of_nodes())
print("Aristas:", G.number_of_edges())
print("Componentes:", nx.number_connected_components(G))

if G.number_of_nodes() > 0:
    print("Componente mayor:", len(max(nx.connected_components(G), key=len)))


# ============================================================
# 9. DETECCIÓN DE COMUNIDADES
# ============================================================

if G.number_of_edges() > 0:
    communities = nx.algorithms.community.greedy_modularity_communities(
        G,
        weight="weight"
    )
else:
    communities = [{n} for n in G.nodes]

communities = sorted(communities, key=len, reverse=True)

node_to_comm = {}

for k, comm in enumerate(communities, start=1):
    for node in comm:
        node_to_comm[node] = k

print("\n==============================")
print("COMUNIDADES DETECTADAS")
print("==============================")

for k, comm in enumerate(communities, start=1):
    print(f"\nComunidad {k} | tamaño: {len(comm)}")

    sub = G.subgraph(comm)
    top = sorted(
        sub.degree(weight="weight"),
        key=lambda x: x[1],
        reverse=True
    )[:12]

    for var, degree in top:
        print(f"{var:8s} | grado ponderado={degree:.2f} | {var_label(var)}")


# Guardar comunidades
comm_rows = []

for k, comm in enumerate(communities, start=1):
    for v in sorted(comm):
        comm_rows.append({
            "community_id": k,
            "variable": v,
            "label": var_label(v)
        })

communities_df = pd.DataFrame(comm_rows)
communities_df.to_csv(OUT_DIR / "comunidades_variables.csv", index=False)


# ============================================================
# 10. TABLA DE ARISTAS
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

print("\n==============================")
print("ARISTAS MÁS FUERTES")
print("==============================")
print(edges_df.head(20))


# ============================================================
# 11. VISUALIZACIÓN LIMPIA DE LA RED DE VARIABLES
#     Con aristas fantasma solo para compactar componentes
# ============================================================

print("\n==============================")
print("GENERANDO RED DE VARIABLES LIMPIA")
print("==============================")

nodes_viz = [n for n in G.nodes() if G.degree(n) > 0]
G_viz = G.subgraph(nodes_viz).copy()

# Para no perder tanta estructura, deja el umbral igual o un poco menor.
# Si subes mucho este valor, el grafo se fragmenta demasiado.
viz_threshold = 0.30

edges_to_remove = [
    (u, v)
    for u, v, d in G_viz.edges(data=True)
    if d["weight"] < viz_threshold
]

G_viz.remove_edges_from(edges_to_remove)
G_viz.remove_nodes_from(list(nx.isolates(G_viz)))

if G_viz.number_of_edges() > 0:

    # --------------------------------------------------------
    # 1. Crear grafo auxiliar SOLO para el layout
    # --------------------------------------------------------

    G_layout = G_viz.copy()

    components = sorted(
        nx.connected_components(G_layout),
        key=len,
        reverse=True
    )

    # Hub de cada componente: nodo con mayor grado ponderado
    component_hubs = []

    for comp in components:
        sub = G_layout.subgraph(comp)
        hub = max(
            sub.nodes(),
            key=lambda n: sub.degree(n, weight="weight")
        )
        component_hubs.append(hub)

    # Conectar componentes con aristas fantasma.
    # Estas aristas NO se dibujan; solo ayudan a compactar el layout.
    if len(component_hubs) > 1:
        main_hub = component_hubs[0]

        for hub in component_hubs[1:]:
            G_layout.add_edge(
                main_hub,
                hub,
                weight=0.18,
                phantom=True
            )

    # --------------------------------------------------------
    # 2. Layout compacto
    # --------------------------------------------------------

    pos = nx.spring_layout(
        G_layout,
        seed=42,
        weight="weight",
        k=0.28,
        iterations=1500,
        scale=1.0
    )

    # Nos quedamos solo con posiciones de nodos reales
    pos = {n: pos[n] for n in G_viz.nodes()}

    # --------------------------------------------------------
    # 3. Colores por comunidad
    # --------------------------------------------------------

    cmap = plt.colormaps.get_cmap("tab20")

    node_colors = [
        cmap((node_to_comm.get(n, 0) % 20) / 20)
        for n in G_viz.nodes()
    ]

    # --------------------------------------------------------
    # 4. Tamaños por grado ponderado
    # --------------------------------------------------------

    weighted_degrees = dict(G_viz.degree(weight="weight"))

    node_sizes = [
        90 + 230 * weighted_degrees.get(n, 0)
        for n in G_viz.nodes()
    ]

    # --------------------------------------------------------
    # 5. Grosor de aristas reales
    # --------------------------------------------------------

    edge_widths = [
        0.35 + 3.2 * G_viz[u][v]["weight"]
        for u, v in G_viz.edges()
    ]

    edge_alphas = [
        min(0.55, 0.12 + G_viz[u][v]["weight"])
        for u, v in G_viz.edges()
    ]

    # --------------------------------------------------------
    # 6. Etiquetas: solo nodos centrales
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7. Dibujar
    # --------------------------------------------------------

    plt.figure(figsize=(12, 9))

    # Dibujar solo aristas reales de G_viz, no las fantasma
    for (u, v), width, alpha in zip(G_viz.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G_viz,
            pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
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

    plt.savefig(
        OUT_DIR / "red_variables_enpscyt_limpia.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.savefig(
        OUT_DIR / "red_variables_enpscyt_limpia.svg",
        bbox_inches="tight"
    )

    plt.close()

    print("Figura guardada: red_variables_enpscyt_limpia.png / .svg")

else:
    print("No hay aristas suficientes para dibujar G_viz.")


# ============================================================
# 12. RED AGREGADA DE COMUNIDADES
# ============================================================

print("\n==============================")
print("GENERANDO RED AGREGADA DE COMUNIDADES")
print("==============================")

# Nombres tentativos. Luego se ajustan leyendo comunidades_variables.csv.
community_names = {
    1: "Capital informativo\ne interés",
    2: "Optimismo\ntecnocientífico",
    3: "Capacidades\nnacionales",
    4: "Prestigio de\ncientíficos",
    5: "Riesgos y\nreservas",
    6: "Participación\ncultural",
    7: "Condiciones\ninstitucionales",
    8: "Satisfacción\ncon el país",
    9: "Religión",
}

H = nx.Graph()

for comm_id in sorted(set(node_to_comm.values())):
    nodes_comm = [
        n for n in G.nodes()
        if node_to_comm.get(n) == comm_id
    ]

    if len(nodes_comm) == 0:
        continue

    H.add_node(
        comm_id,
        size=len(nodes_comm),
        label=community_names.get(comm_id, f"Comunidad {comm_id}")
    )

edge_weights = defaultdict(float)
edge_counts = defaultdict(int)

for u, v, d in G.edges(data=True):
    cu = node_to_comm.get(u)
    cv = node_to_comm.get(v)

    if cu is None or cv is None:
        continue

    if cu == cv:
        continue

    a, b = sorted([cu, cv])

    edge_weights[(a, b)] += d["weight"]
    edge_counts[(a, b)] += 1

for (a, b), total_weight in edge_weights.items():
    H.add_edge(
        a,
        b,
        weight=total_weight,
        count=edge_counts[(a, b)]
    )

if H.number_of_edges() > 0:
    pos_H = nx.spring_layout(
        H,
        seed=7,
        weight="weight",
        k=1.4,
        iterations=700
    )

    node_sizes_H = [
        900 + 230 * H.nodes[n]["size"]
        for n in H.nodes()
    ]

    edge_widths_H = [
        0.7 + 0.18 * H[u][v]["weight"]
        for u, v in H.edges()
    ]

    labels_H = {
        n: H.nodes[n]["label"]
        for n in H.nodes()
    }

    cmap = plt.colormaps.get_cmap("tab20")

    node_colors_H = [
        cmap((n % 20) / 20)
        for n in H.nodes()
    ]

    plt.figure(figsize=(12, 9))

    nx.draw_networkx_edges(
        H,
        pos_H,
        width=edge_widths_H,
        alpha=0.35,
        edge_color="gray"
    )

    nx.draw_networkx_nodes(
        H,
        pos_H,
        node_size=node_sizes_H,
        node_color=node_colors_H,
        linewidths=1.2,
        edgecolors="white",
        alpha=0.95
    )

    nx.draw_networkx_labels(
        H,
        pos_H,
        labels=labels_H,
        font_size=10,
        font_weight="bold"
    )

    plt.title(
        "Arquitectura relacional de la cultura científica en el Perú",
        fontsize=16,
        pad=20
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "red_comunidades_enpscyt_limpia.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.savefig(
        OUT_DIR / "red_comunidades_enpscyt_limpia.svg",
        bbox_inches="tight"
    )

    plt.close()

    print("Figura guardada: red_comunidades_enpscyt_limpia.png / .svg")
else:
    print("No hay aristas suficientes para dibujar H.")


# ============================================================
# 13. CONVERTIR COMUNIDADES EN ÍNDICES INDIVIDUALES
# ============================================================

print("\n==============================")
print("CONSTRUYENDO ÍNDICES DE COMUNIDAD")
print("==============================")

Z_dict = {}

for c in X.columns:
    mu, sd = weighted_mean_std(X[c], w)

    if sd > 0:
        Z_dict[c] = (X[c] - mu) / sd

Z = pd.DataFrame(Z_dict, index=df.index)

scores = pd.DataFrame(index=df.index)
community_vars = {}

for k, comm in enumerate(communities, start=1):
    vars_k = [v for v in comm if v in Z.columns]

    # Usamos solo comunidades con tamaño razonable
    if len(vars_k) >= 3:
        name = f"com_{k:02d}"
        scores[name] = Z[vars_k].mean(axis=1)
        community_vars[name] = vars_k

print("Índices creados:", scores.shape)
print(scores.head())

# Guardar composición de índices
index_rows = []

for name, vars_k in community_vars.items():
    for v in vars_k:
        index_rows.append({
            "index": name,
            "variable": v,
            "label": var_label(v)
        })

indices_df = pd.DataFrame(index_rows)
indices_df.to_csv(OUT_DIR / "indices_comunidades_variables.csv", index=False)


# ============================================================
# 14. VARIABLE DEPENDIENTE: APOYO A AUMENTAR INVERSIÓN
# ============================================================

print("\n==============================")
print("VARIABLE DEPENDIENTE")
print("==============================")

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

model_df = scores.copy()
model_df["y_apoya_aumento"] = y
model_df["Fexp"] = w
model_df["w_norm"] = model_df["Fexp"] / model_df["Fexp"].mean()

print(pd.Series(model_df["y_apoya_aumento"]).value_counts(dropna=False))


# ============================================================
# 15. CONTROLES SOCIODEMOGRÁFICOS
# ============================================================

control_candidates = [
    "P41",          # sexo
    "P42",          # edad
    "p43_grupo",    # nivel educativo
    "mod0_5",       # NSE
    "mod0_1_dep"    # departamento
]

controls = [c for c in control_candidates if c in df.columns]

for c in controls:
    model_df[c] = df[c]

print("\nControles usados:")
print(controls)


# ============================================================
# 16. MODELO LOGÍSTICO PONDERADO
# ============================================================

print("\n==============================")
print("MODELO LOGÍSTICO PONDERADO")
print("==============================")

score_cols = scores.columns.tolist()

if len(score_cols) == 0:
    raise ValueError(
        "No se crearon índices de comunidad. "
        "Baja el threshold o revisa la recodificación."
    )

model_vars = [
    "y_apoya_aumento",
    "w_norm"
] + controls + score_cols

m = model_df[model_vars].dropna().copy()

print("Casos usados en modelo:", len(m))

X_design = pd.get_dummies(
    m[controls + score_cols],
    drop_first=True
)

X_design = sm.add_constant(X_design).astype(float)

y_model = m["y_apoya_aumento"].astype(float)
weights_model = m["w_norm"].astype(float)

model = sm.GLM(
    y_model,
    X_design,
    family=sm.families.Binomial(),
    freq_weights=weights_model
)

res = model.fit(cov_type="HC1")

print(res.summary())


# ============================================================
# 17. ODDS RATIOS DE ÍNDICES DE RED
# ============================================================

coef = res.params[score_cols]
ci = res.conf_int().loc[score_cols]

or_table = pd.DataFrame({
    "OR": np.exp(coef),
    "IC95_inf": np.exp(ci[0]),
    "IC95_sup": np.exp(ci[1]),
    "p_value": res.pvalues[score_cols],
})

or_table = or_table.sort_values("OR", ascending=False)

or_table.to_csv(OUT_DIR / "modelo_odds_ratios_indices_red.csv", index=True)

print("\n==============================")
print("ODDS RATIOS")
print("==============================")
print(or_table)


# ============================================================
# 18. COMPARACIÓN MODELO BASE VS MODELO CON ÍNDICES
# ============================================================

print("\n==============================")
print("COMPARACIÓN DE MODELOS")
print("==============================")

def fit_logit_weighted(data, predictors):
    X_tmp = pd.get_dummies(data[predictors], drop_first=True)
    X_tmp = sm.add_constant(X_tmp).astype(float)

    y_tmp = data["y_apoya_aumento"].astype(float)
    w_tmp = data["w_norm"].astype(float)

    mod = sm.GLM(
        y_tmp,
        X_tmp,
        family=sm.families.Binomial(),
        freq_weights=w_tmp
    )

    return mod.fit(cov_type="HC1")


base_data = model_df[
    ["y_apoya_aumento", "w_norm"] + controls + score_cols
].dropna().copy()

res_base = fit_logit_weighted(base_data, controls)
res_full = fit_logit_weighted(base_data, controls + score_cols)

comparison = pd.DataFrame({
    "modelo": [
        "Base sociodemográfico",
        "Base + índices de red"
    ],
    "AIC": [
        res_base.aic,
        res_full.aic
    ],
    "LogLik": [
        res_base.llf,
        res_full.llf
    ],
    "Df_model": [
        res_base.df_model,
        res_full.df_model
    ],
    "N": [
        len(base_data),
        len(base_data)
    ]
})

comparison["delta_AIC_vs_base"] = comparison["AIC"] - comparison.loc[0, "AIC"]

comparison.to_csv(OUT_DIR / "comparacion_modelos.csv", index=False)

print(comparison)


# ============================================================
# 19. GUARDAR BASE DE ÍNDICES
# ============================================================

scores_out = scores.copy()
scores_out["y_apoya_aumento"] = model_df["y_apoya_aumento"]
scores_out["Fexp"] = model_df["Fexp"]

for c in controls:
    scores_out[c] = model_df[c]

scores_out.to_csv(OUT_DIR / "base_indices_para_modelo.csv", index=False)


# ============================================================
# 20. RESUMEN FINAL
# ============================================================

print("\n==============================")
print("RESUMEN FINAL")
print("==============================")
print(f"Título tentativo:")
print(
    "Arquitectura relacional de la cultura científica en el Perú: "
    "redes de información, actitudes tecnológicas y apoyo ciudadano "
    "a la inversión pública en ciencia y tecnología"
)

print("\nArchivos guardados en:")
print(OUT_DIR)

print("\nPrincipales salidas:")
print("- matriz_correlaciones_ponderadas.csv")
print("- red_aristas_variables.csv")
print("- comunidades_variables.csv")
print("- indices_comunidades_variables.csv")
print("- red_variables_enpscyt_limpia.png / .svg")
print("- red_comunidades_enpscyt_limpia.png / .svg")
print("- modelo_odds_ratios_indices_red.csv")
print("- comparacion_modelos.csv")
print("- base_indices_para_modelo.csv")