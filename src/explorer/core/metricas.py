"""Tablas de métricas a partir de las predicciones del modelo."""
import numpy as np
import pandas as pd


def tabla_predicciones(meta, res, config):
    """Un DataFrame por imagen. Imágenes con países fuera del modelo quedan con en_modelo=False."""
    paises = np.asarray(config["paises"])
    p2c = config["pais_a_continente"]
    pais_a_idx = {p: i for i, p in enumerate(paises)}
    probs = res["probs"].astype(np.float32)
    y_true = meta["country_code"].map(pais_a_idx).fillna(-1).astype(int).to_numpy()
    y_pred = probs.argmax(1)
    validos = y_true >= 0
    fila = np.arange(len(meta))
    p_real = np.where(validos, probs[fila, np.where(validos, y_true, 0)], np.nan)
    rank = np.where(validos, (probs > np.nan_to_num(p_real)[:, None]).sum(1) + 1, 0)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    df = meta.copy()
    df["pais_real"] = meta["country_code"]
    df["pais_predicho"] = paises[y_pred]
    df["continente_real"] = df["pais_real"].map(p2c)
    df["continente_predicho"] = df["pais_predicho"].map(p2c)
    df["confianza"] = probs.max(1)
    df["prob_real"] = p_real
    df["rank_real"] = rank
    df["top5"] = [" ".join(paises[t]) for t in top5]
    df["acierto_pais"] = validos & (y_true == y_pred)
    df["acierto_continente"] = validos & (df["continente_real"] == df["continente_predicho"])
    df["acierto_top5"] = validos & (rank <= 5)
    df["en_modelo"] = validos
    return df


def _top_columnas(serie, k=3):
    total = serie.sum()
    if total == 0:
        return ""
    return ", ".join(f"{p} {v / total:.0%}" for p, v in serie.nlargest(k).items() if v > 0)


def resumen_por_pais(df, config):
    paises = config["paises"]
    d = df[df["en_modelo"]]
    cm = pd.crosstab(d["pais_real"], d["pais_predicho"]).reindex(
        index=paises, columns=paises, fill_value=0)
    n, pred = cm.sum(axis=1), cm.sum(axis=0)
    tp = pd.Series(np.diag(cm.to_numpy()), index=paises)
    recall = tp / n.replace(0, np.nan)
    precision = tp / pred.replace(0, np.nan)
    f1 = 2 * precision * recall / (precision + recall)
    off_valores = cm.to_numpy().copy()
    np.fill_diagonal(off_valores, 0)
    off = pd.DataFrame(off_valores, index=cm.index, columns=cm.columns)
    conf_ok = d[d["acierto_pais"]].groupby("pais_real")["confianza"].mean()
    conf_err = d[~d["acierto_pais"]].groupby("pais_real")["confianza"].mean()
    out = pd.DataFrame({
        "continente": [config["pais_a_continente"][p] for p in paises],
        "n": n, "n_predichas": pred, "recall": recall, "precision": precision, "f1": f1,
        "top5": d.groupby("pais_real")["acierto_top5"].mean().reindex(paises),
        "conf_aciertos": conf_ok.reindex(paises), "conf_errores": conf_err.reindex(paises),
        "se_confunde_con": [_top_columnas(off.loc[p]) for p in paises],
        "falsos_positivos_de": [_top_columnas(off[p]) for p in paises],
    }, index=paises)
    out.index.name = "pais"
    return out


def resumen_por_continente(df):
    d = df[df["en_modelo"]]
    return d.groupby("continente_real").agg(
        n=("acierto_pais", "size"), acc_pais=("acierto_pais", "mean"),
        acc_continente=("acierto_continente", "mean"), top5=("acierto_top5", "mean"))


def matriz_confusion(df, config, nivel="pais"):
    d = df[df["en_modelo"]]
    if nivel == "pais":
        etiquetas, r, p = config["paises"], "pais_real", "pais_predicho"
    else:
        etiquetas = sorted(set(config["pais_a_continente"].values()))
        r, p = "continente_real", "continente_predicho"
    return pd.crosstab(d[r], d[p]).reindex(index=etiquetas, columns=etiquetas, fill_value=0)


def pares_confusion(df, config, min_n=1):
    d = df[df["en_modelo"] & ~df["acierto_pais"]]
    totales = df[df["en_modelo"]].groupby("pais_real").size()
    pares = d.groupby(["pais_real", "pais_predicho"]).size().rename("n_errores").reset_index()
    pares["n_real"] = pares["pais_real"].map(totales)
    pares["pct_de_real"] = pares["n_errores"] / pares["n_real"]
    p2c = config["pais_a_continente"]
    pares["mismo_continente"] = pares["pais_real"].map(p2c) == pares["pais_predicho"].map(p2c)
    return pares[pares["n_errores"] >= min_n].sort_values("n_errores", ascending=False)
