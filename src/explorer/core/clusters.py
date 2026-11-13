"""Prototipos de los clasificadores por cluster: asignación, distancias y geometría para mapas."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import SphericalVoronoi

from .paths import resolver_ruta

RADIO_TIERRA_KM = 6371.0
# Umbrales habituales de geolocalización (calle, ciudad, región, país, continente).
UMBRALES_KM = (1, 25, 200, 750, 2500)


def a_xyz(lat, lon):
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _a_latlon(xyz):
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1))), np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def asignar(lat, lon, protos):
    """Cluster del prototipo más cercano (mayor producto punto = menor distancia angular),
    igual que en el muestreo y el entrenamiento. `protos` es (K, 2) con lat, lon."""
    return (a_xyz(lat, lon) @ a_xyz(protos[:, 0], protos[:, 1]).T).argmax(axis=1)


def distancias_entre_prototipos(protos):
    return haversine_km(protos[:, None, 0], protos[:, None, 1], protos[None, :, 0], protos[None, :, 1])


def cargar_prototipos(config, carpeta_corrida):
    """DataFrame (ind_cluster, lat, lon) ordenado. Prioriza la copia guardada junto a la corrida."""
    copia = Path(carpeta_corrida) / "prototipos.csv"
    info = config.get("prototipos", {})
    ruta = copia if copia.is_file() else resolver_ruta(info.get("ruta"))
    if ruta is None:
        raise FileNotFoundError(f"No se encontraron los prototipos de la corrida ({info.get('ruta')}).")
    aviso = None
    if ruta != copia and info.get("sha256"):
        if hashlib.sha256(ruta.read_bytes()).hexdigest() != info["sha256"]:
            aviso = f"Los prototipos de {ruta.name} no coinciden con el sha256 del entrenamiento."
    protos = pd.read_csv(ruta).sort_values("ind_cluster").reset_index(drop=True)
    if not (protos["ind_cluster"].to_numpy() == np.arange(len(protos))).all():
        raise ValueError(f"{ruta}: ind_cluster no es 0..K-1.")
    if len(protos) != int(config["num_clases"]):
        raise ValueError(f"{ruta}: {len(protos)} prototipos pero el modelo tiene {config['num_clases']} clases.")
    return protos[["ind_cluster", "lat", "lon"]], aviso


def describir_clusters(protos, meta_ref, pais_a_continente):
    """Nombre legible y composición de cada cluster según un conjunto de referencia (el train)."""
    coords = protos[["lat", "lon"]].to_numpy(float)
    ok = meta_ref[["latitude", "longitude"]].notna().all(axis=1).to_numpy()
    ref = meta_ref.loc[ok]
    cluster = asignar(ref["latitude"].to_numpy(float), ref["longitude"].to_numpy(float), coords)
    conteos = pd.crosstab(cluster, ref["country_code"].fillna("?").to_numpy()).reindex(
        range(len(protos)), fill_value=0)
    filas = []
    for k, fila in conteos.iterrows():
        total = int(fila.sum())
        top = fila[fila > 0].sort_values(ascending=False)
        dominante = top.index[0] if total else None
        filas.append({
            "ind_cluster": k, "lat": coords[k, 0], "lon": coords[k, 1],
            "etiqueta": f"C{k:02d} {dominante}" if dominante else f"C{k:02d}",
            "pais_dominante": dominante, "pct_dominante": top.iloc[0] / total if total else np.nan,
            "paises": " · ".join(f"{p} {v / total:.0%}" for p, v in top.head(4).items()),
            "continente": pais_a_continente.get(dominante, "Sin datos"), "n_referencia": total,
        })
    return pd.DataFrame(filas).set_index("ind_cluster")


# ---------- geometría para dibujar ----------

def _arco(a, b, paso_grados=1.0):
    """Puntos sobre el círculo máximo entre los vectores unitarios a y b."""
    theta = np.arccos(np.clip(a @ b, -1, 1))
    if theta < 1e-9:
        return np.stack([a, b])
    t = np.linspace(0, 1, max(2, int(np.ceil(np.degrees(theta) / paso_grados)) + 1))[:, None]
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def _cortar_antimeridiano(lat, lon):
    """Listas lat/lon con None donde la línea cruza ±180°, interpolando el punto de corte."""
    out_lat, out_lon = [float(lat[0])], [float(lon[0])]
    for i in range(1, len(lat)):
        la0, lo0, la1, lo1 = lat[i - 1], lon[i - 1], lat[i], lon[i]
        if abs(lo1 - lo0) > 180:
            borde = 180.0 if lo0 > 0 else -180.0
            lo1_cont = lo1 + 360 if lo0 > 0 else lo1 - 360
            t = (borde - lo0) / (lo1_cont - lo0) if lo1_cont != lo0 else 0.5
            la_c = float(la0 + t * (la1 - la0))
            out_lat += [la_c, None, la_c]
            out_lon += [borde, None, -borde]
        out_lat.append(float(la1))
        out_lon.append(float(lo1))
    return out_lat, out_lon


def _voronoi(protos):
    sv = SphericalVoronoi(a_xyz(protos[:, 0], protos[:, 1]), radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    return sv


def aristas_voronoi(protos):
    """Bordes de todas las celdas como una sola traza (listas lat/lon separadas por None)."""
    sv = _voronoi(np.asarray(protos, float))
    vistas = set()
    lats, lons = [], []
    for region in sv.regions:
        for a, b in zip(region, region[1:] + region[:1]):
            clave = (min(a, b), max(a, b))
            if clave in vistas:
                continue
            vistas.add(clave)
            la, lo = _a_latlon(_arco(sv.vertices[a], sv.vertices[b]))
            la, lo = _cortar_antimeridiano(la, lo)
            lats += la + [None]
            lons += lo + [None]
    return lats, lons


def contornos_voronoi(protos):
    """Por cluster: (lats, lons, cruza_antimeridiano) del contorno cerrado de su celda."""
    sv = _voronoi(np.asarray(protos, float))
    contornos = []
    for region in sv.regions:
        puntos = np.concatenate([_arco(sv.vertices[a], sv.vertices[b])[:-1]
                                 for a, b in zip(region, region[1:] + region[:1])])
        puntos = np.vstack([puntos, puntos[:1]])
        la, lo = _a_latlon(puntos)
        la2, lo2 = _cortar_antimeridiano(la, lo)
        contornos.append((la2, lo2, None in la2))
    return contornos
