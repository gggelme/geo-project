"""Mapas Plotly (geo) para los clasificadores por cluster."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from explorer.core.clusters import _a_latlon, _arco, _cortar_antimeridiano, a_xyz

# 74 colores distinguibles: alcanza para K=70 sin repetir.
PALETA = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
PALETA_CONTINENTES = dict(zip(
    ["África", "América del Norte", "América del Sur", "Asia", "Europa", "Oceanía", "Sin datos"],
    px.colors.qualitative.Set2))
ESCALAS = {"recall": "RdYlGn", "precision": "RdYlGn", "f1": "RdYlGn", "top5": "RdYlGn",
           "error_mediano_km": "RdYlGn_r", "cota_mediana_km": "RdYlGn_r", "n": "Blues"}


def color_cluster(k):
    return PALETA[int(k) % len(PALETA)]


def figura(alto=600, centro=None, escala=1.0, revision=None):
    fig = go.Figure()
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#b8b8b8",
                    countrywidth=0.5, showcoastlines=True, coastlinecolor="#9a9a9a", showland=True,
                    landcolor="#f3f1ec", showocean=True, oceancolor="#e4eef6", showframe=False,
                    lataxis_range=[-60, 85], projection_scale=escala)
    if centro is not None:
        fig.update_geos(center=dict(lat=centro[0], lon=centro[1]), projection_rotation=dict(lon=centro[1]),
                        lataxis_range=None)
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=0, b=0), uirevision=revision,
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0, bgcolor="rgba(255,255,255,0.7)"))
    return fig


def bordes(fig, aristas, color="#555", ancho=0.8):
    fig.add_trace(go.Scattergeo(lat=aristas[0], lon=aristas[1], mode="lines", hoverinfo="skip",
                                line=dict(color=color, width=ancho), showlegend=False, name="bordes"))


def contorno(fig, contorno_cluster, color="#d62728", ancho=3, relleno=True):
    lats, lons, cruza = contorno_cluster
    usar_relleno = relleno and not cruza  # el relleno se rompe si el polígono cruza ±180°
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", hoverinfo="skip", showlegend=False,
                                line=dict(color=color, width=ancho),
                                fill="toself" if usar_relleno else None,
                                fillcolor="rgba(214,39,40,0.10)" if usar_relleno else None))


def geodesica(lat1, lon1, lat2, lon2):
    pts = _arco(a_xyz(lat1, lon1), a_xyz(lat2, lon2))
    return _cortar_antimeridiano(*_a_latlon(pts))


def flujos(fig, origen, destinos, color, nombre):
    """Líneas desde `origen` (lat, lon) a cada destino (DataFrame lat, lon, n, texto)."""
    if destinos.empty:
        return
    maximo = destinos["n"].max()
    for i, (_, f) in enumerate(destinos.iterrows()):
        la, lo = geodesica(origen[0], origen[1], f["lat"], f["lon"])
        fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", hoverinfo="skip", opacity=0.75,
                                    line=dict(color=color, width=1.5 + 7 * f["n"] / maximo),
                                    name=nombre, legendgroup=nombre, showlegend=i == 0))
    fig.add_trace(go.Scattergeo(lat=destinos["lat"], lon=destinos["lon"], mode="markers",
                                marker=dict(size=9, color=color, line=dict(color="white", width=1)),
                                hovertext=destinos["texto"], hoverinfo="text", name=nombre,
                                legendgroup=nombre, showlegend=False))


def puntos_imagenes(fig, d, colores, nombre, tamano=4, opacidad=0.55, mostrar_leyenda=False):
    fig.add_trace(go.Scattergeo(
        lat=d["latitude"], lon=d["longitude"], mode="markers", name=nombre, showlegend=mostrar_leyenda,
        marker=dict(size=tamano, color=colores, opacity=opacidad, line=dict(width=0)),
        customdata=np.asarray(d["clase_real"]), hoverinfo="text",
        hovertext=[f"{r} → {p}<br>{c} · {e:,.0f} km" for r, p, c, e in
                   zip(d["clase_real"], d["clase_predicha"], d["pais_real"], d["error_km"])]))


def _hover(tabla):
    t = tabla.fillna({"n": 0})
    return [f"<b>{e}</b><br>{f['paises']}<br>n={f['n']:.0f} · recall {f['recall']:.0%} · "
            f"precisión {f['precision']:.0%}<br>error mediano {f['error_mediano_km']:,.0f} km"
            for e, f in t.iterrows()]


def _traza_prototipos(sub, marcador, nombre, etiquetas, leyenda):
    return go.Scattergeo(
        lat=sub["lat"], lon=sub["lon"], mode="markers+text" if etiquetas else "markers",
        text=[e.split(" ")[0] for e in sub.index], textposition="top center",
        textfont=dict(size=10, color="#222"), customdata=np.asarray(sub.index),
        hovertext=_hover(sub), hoverinfo="text", marker=marcador, name=nombre, showlegend=leyenda)


def prototipos(fig, tabla, color_por, etiquetas=True, resaltado=None):
    """Un marcador por cluster. `tabla` es resumen_por_cluster (índice = etiqueta)."""
    def tam(sub):
        return np.where(sub.index == resaltado, 18, 11)

    borde = dict(color="#222", width=1)
    if color_por == "continente":
        for cont, sub in tabla.groupby("continente"):
            fig.add_trace(_traza_prototipos(
                sub, dict(size=tam(sub), color=PALETA_CONTINENTES.get(cont, "#888"), line=borde),
                cont, etiquetas, True))
        return
    if color_por == "cluster":
        marcador = dict(size=tam(tabla), color=[color_cluster(k) for k in tabla["ind_cluster"]], line=borde)
    else:
        marcador = dict(size=tam(tabla), color=tabla[color_por], colorscale=ESCALAS.get(color_por, "Viridis"),
                        line=borde, showscale=True, colorbar=dict(title=color_por, thickness=12, len=0.6))
        if color_por in ("recall", "precision", "f1", "top5"):
            marcador |= dict(cmin=0, cmax=1)
    fig.add_trace(_traza_prototipos(tabla, marcador, "prototipos", etiquetas, False))
