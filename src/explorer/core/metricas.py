"""Tablas de métricas a partir de las predicciones del modelo.

Columnas comunes a ambas tareas: clase_real, clase_predicha (nombres de `clases`), acierto,
acierto_top5, acierto_continente, confianza, prob_real, rank_real, top5 y en_modelo.
Por país se agregan pais_predicho/acierto_pais; por cluster, cluster_real/cluster_predicho,
coordenadas del prototipo predicho y errores en km.
"""
import numpy as np
import pandas as pd

from .clusters import UMBRALES_KM, asignar, distancias_entre_prototipos, haversine_km


def _comunes(df, probs, y_true, y_pred, validos, clases):
    clases = np.asarray(clases, dtype=object)
    fila = np.arange(len(df))
    p_real = np.where(validos, probs[fila, np.where(validos, y_true, 0)], np.nan)
    rank = np.where(validos, (probs > np.nan_to_num(p_real)[:, None]).sum(1) + 1, 0)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    df["clase_real"] = np.where(validos, clases[np.where(validos, y_true, 0)], None)
    df["clase_predicha"] = clases[y_pred]
    df["confianza"] = probs.max(1)
    df["prob_real"] = p_real
    df["rank_real"] = rank
    df["top5"] = [" | ".join(clases[t]) for t in top5]
    df["acierto"] = validos & (y_true == y_pred)
    df["acierto_top5"] = validos & (rank <= 5)
    df["en_modelo"] = validos


def tabla_predicciones(meta, res, config, clusters=None):
    """Un DataFrame por imagen. Con `clusters` (de describir_clusters) arma la tabla por cluster."""
    probs = res["probs"].astype(np.float32)
    y_pred = probs.argmax(1)
    p2c = config["pais_a_continente"]
    df = meta.copy()
    df["pais_real"] = meta["country_code"]
    if clusters is None:
        paises = list(config["paises"])
        y_true = meta["country_code"].map({p: i for i, p in enumerate(paises)}).fillna(-1).astype(int).to_numpy()
        validos = y_true >= 0
        _comunes(df, probs, y_true, y_pred, validos, paises)
        df["pais_predicho"] = df["clase_predicha"]
        df["continente_real"] = df["pais_real"].map(p2c)
        df["continente_predicho"] = df["pais_predicho"].map(p2c)
        df["acierto_pais"] = df["acierto"]
    else:
        coords = clusters[["lat", "lon"]].to_numpy(float)
        lat, lon = meta["latitude"].to_numpy(float), meta["longitude"].to_numpy(float)
        validos = np.isfinite(lat) & np.isfinite(lon)
        y_true = np.full(len(meta), -1)
        y_true[validos] = asignar(lat[validos], lon[validos], coords)
        _comunes(df, probs, y_true, y_pred, validos, list(clusters["etiqueta"]))
        # Predicción de mínimo riesgo de Bayes: la celda con menor distancia esperada.
        y_mbr = (probs @ distancias_entre_prototipos(coords)).argmin(1)
        cont_cluster = clusters["continente"].to_numpy(dtype=object)
        df["cluster_real"] = y_true
        df["cluster_predicho"] = y_pred
        df["lat_predicha"], df["lon_predicha"] = coords[y_pred, 0], coords[y_pred, 1]
        df["error_km"] = haversine_km(lat, lon, coords[y_pred, 0], coords[y_pred, 1])
        df["error_km_mbr"] = haversine_km(lat, lon, coords[y_mbr, 0], coords[y_mbr, 1])
        # Cota: error si siempre se acertara la celda (distancia de la foto a su prototipo).
        t = np.where(validos, y_true, 0)
        df["error_km_cota"] = np.where(validos, haversine_km(lat, lon, coords[t, 0], coords[t, 1]), np.nan)
        df["acierto_mbr"] = validos & (y_mbr == y_true)
        df["continente_real"] = df["pais_real"].map(p2c).fillna(
            pd.Series(np.where(validos, cont_cluster[t], None), index=df.index))
        df["continente_predicho"] = cont_cluster[y_pred]
        df["pais_predicho"] = clusters["pais_dominante"].to_numpy(dtype=object)[y_pred]
        df["acierto_pais"] = validos & (df["pais_real"] == df["pais_predicho"])
    df["acierto_continente"] = validos & (df["continente_real"] == df["continente_predicho"])
    return df


def resumen_km(errores):
    """Mediana, media y % de imágenes dentro de cada umbral."""
    e = np.asarray(errores, float)
    e = e[np.isfinite(e)]
    out = {"mediana_km": np.median(e) if len(e) else np.nan, "media_km": e.mean() if len(e) else np.nan}
    for u in UMBRALES_KM:
        out[f"<= {u:,} km"] = (e <= u).mean() if len(e) else np.nan
    return out


def _top_columnas(serie, k=3):
    total = serie.sum()
    if total == 0:
        return ""
    return ", ".join(f"{p} {v / total:.0%}" for p, v in serie.nlargest(k).items() if v > 0)


def resumen_por_clase(df, clases):
    d = df[df["en_modelo"]]
    cm = pd.crosstab(d["clase_real"], d["clase_predicha"]).reindex(index=clases, columns=clases, fill_value=0)
    n, pred = cm.sum(axis=1), cm.sum(axis=0)
    tp = pd.Series(np.diag(cm.to_numpy()), index=clases)
    recall = tp / n.replace(0, np.nan)
    precision = tp / pred.replace(0, np.nan)
    off_valores = cm.to_numpy().copy()
    np.fill_diagonal(off_valores, 0)
    off = pd.DataFrame(off_valores, index=cm.index, columns=cm.columns)
    g = d.groupby("clase_real")
    out = pd.DataFrame({
        "n": n, "n_predichas": pred, "recall": recall, "precision": precision,
        "f1": 2 * precision * recall / (precision + recall),
        "top5": g["acierto_top5"].mean().reindex(clases),
        "conf_aciertos": d[d["acierto"]].groupby("clase_real")["confianza"].mean().reindex(clases),
        "conf_errores": d[~d["acierto"]].groupby("clase_real")["confianza"].mean().reindex(clases),
        "se_confunde_con": [_top_columnas(off.loc[c]) for c in clases],
        "falsos_positivos_de": [_top_columnas(off[c]) for c in clases],
    }, index=pd.Index(clases, name="clase"))
    if "error_km" in d:
        out.insert(5, "error_mediano_km", g["error_km"].median().reindex(clases))
        out.insert(6, "cota_mediana_km", g["error_km_cota"].median().reindex(clases))
    return out


def resumen_por_pais(df, config):
    out = resumen_por_clase(df, list(config["paises"]))
    out.insert(0, "continente", [config["pais_a_continente"][p] for p in out.index])
    out.index.name = "pais"
    return out


def resumen_por_cluster(df, clusters):
    out = resumen_por_clase(df, list(clusters["etiqueta"]))
    info = clusters.set_index("etiqueta")[["continente", "paises", "lat", "lon"]]
    out = info.join(out)
    out.insert(0, "ind_cluster", clusters.index.to_numpy())
    out.index.name = "cluster"
    return out


def resumen_por_grupo(df, columna):
    """Métricas agregadas por una columna (continente_real, pais_real, ...)."""
    d = df[df["en_modelo"]]
    agg = {"n": ("acierto", "size"), "acc": ("acierto", "mean"), "top5": ("acierto_top5", "mean"),
           "acc_continente": ("acierto_continente", "mean")}
    if "error_km" in d:
        agg |= {"acc_pais": ("acierto_pais", "mean"), "error_mediano_km": ("error_km", "median"),
                "cota_mediana_km": ("error_km_cota", "median")}
    return d.groupby(columna).agg(**agg)


def resumen_por_continente(df):
    return resumen_por_grupo(df, "continente_real")


def matriz_confusion(df, etiquetas, nivel="clase"):
    d = df[df["en_modelo"]]
    r, p = ("clase_real", "clase_predicha") if nivel == "clase" else ("continente_real", "continente_predicho")
    return pd.crosstab(d[r], d[p]).reindex(index=etiquetas, columns=etiquetas, fill_value=0)


def pares_confusion(df, min_n=1, clusters=None):
    d = df[df["en_modelo"] & ~df["acierto"]]
    totales = df[df["en_modelo"]].groupby("clase_real").size()
    pares = d.groupby(["clase_real", "clase_predicha"]).agg(
        n_errores=("acierto", "size"),
        mismo_continente=("acierto_continente", "mean")).reset_index()
    pares["mismo_continente"] = pares["mismo_continente"] > 0.5
    pares["n_real"] = pares["clase_real"].map(totales)
    pares["pct_de_real"] = pares["n_errores"] / pares["n_real"]
    if clusters is not None:
        pos = clusters.reset_index().set_index("etiqueta")
        a, b = pos.loc[pares["clase_real"]], pos.loc[pares["clase_predicha"]]
        pares["distancia_km"] = haversine_km(a["lat"].to_numpy(), a["lon"].to_numpy(),
                                             b["lat"].to_numpy(), b["lon"].to_numpy())
        pares["mismo_continente"] = a["continente"].to_numpy() == b["continente"].to_numpy()
    return pares[pares["n_errores"] >= min_n].sort_values("n_errores", ascending=False)
